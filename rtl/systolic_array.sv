`timescale 1ns/1ps

// =============================================================================
// 模块名称：systolic_array
//
// 功能说明：
//   本模块实现一个可参数化的二维脉动阵列。默认配置为 8×8 个物理 MAC，
//   每个 MAC 同时接收两个激活 x0/x1 和两个权重 w0/w1，并在内部维护四路
//   累加结果。因此，默认 8×8 物理阵列在逻辑上对应一个 16×16 INT8 阵列。
//
// 数据流方向：
//   1. 激活 X 从阵列左侧进入，并在每个 MAC 中打一拍后向右传播；
//   2. 权重 W 从阵列上侧进入，并在每个 MAC 中打一拍后向下传播；
//   3. 位于物理坐标 (row, column) 的 MAC 负责逻辑输出中的一个 2×2 块：
//        result_00 -> 逻辑坐标 (2*row,   2*column)
//        result_01 -> 逻辑坐标 (2*row,   2*column+1)
//        result_10 -> 逻辑坐标 (2*row+1, 2*column)
//        result_11 -> 逻辑坐标 (2*row+1, 2*column+1)
//
// Align 原理：
//   为了使相同 K 下标的 X 和 W 在目标 MAC 同一拍相遇，阵列入口必须将数据
//   错拍成斜向波前：第 row 个 X 输入延迟 row 拍，第 column 个 W 输入延迟
//   column 拍。经过 Align 和 MAC 级间寄存后，同一 K 项到达 MAC(row,column)
//   的总延迟均为 row+column 拍。
//
// 输出说明：
//   当前接口只导出每个物理列最底部 MAC 的四路结果，同时导出该列底部的
//   w_last。w_last_o[column] 拉高时，该列底部 MAC 已在同一时钟沿完成最后一项
//   累加，因此对应的四路 result 输出有效。完整阵列的行选择 MUX 和输出 Buffer
//   将在更高层模块中实现。
// =============================================================================
module systolic_array #(
    // 单个激活/权重元素的位宽，当前默认使用 signed INT8。
    parameter int DATA_WIDTH = 8,
    // 每一路点积累加器的位宽，默认使用 signed INT32。
    parameter int ACC_WIDTH  = 32,
    // 物理阵列的行数和列数；当前设计使用方阵。
    parameter int ARRAY_SIZE = 8
) (
    input  logic                                         clk,
    // 低电平有效同步复位，传递给 Align 寄存器和所有 MAC。
    input  logic                                         rst_n,

    // -------------------------------------------------------------------------
    // 激活输入：每个物理行输入一对 signed 数据 x0/x1。
    // valid/first/last 是所有物理行共享的标量控制信号，在 Align 内部分发到
    // 各行延迟链；first/last 分别标记当前 K 序列的首项和末项。
    // -------------------------------------------------------------------------
    input  logic                                         x_valid_i,
    input  logic                                         x_first_i,
    input  logic                                         x_last_i,
    input  logic signed [ARRAY_SIZE-1:0][DATA_WIDTH-1:0] x0_i,
    input  logic signed [ARRAY_SIZE-1:0][DATA_WIDTH-1:0] x1_i,

    // -------------------------------------------------------------------------
    // 权重输入：每个物理列输入一对 signed 数据 w0/w1。
    // valid/first/last 是所有物理列共享的标量控制信号，在 Align 内部分发到
    // 各列延迟链。正常工作时，同一 K 项的 X 和 W 应使用一致的控制模式。
    // -------------------------------------------------------------------------
    input  logic                                         w_valid_i,
    input  logic                                         w_first_i,
    input  logic                                         w_last_i,
    input  logic signed [ARRAY_SIZE-1:0][DATA_WIDTH-1:0] w0_i,
    input  logic signed [ARRAY_SIZE-1:0][DATA_WIDTH-1:0] w1_i,

    // -------------------------------------------------------------------------
    // 列底输出：索引 column 对应物理 MAC(ARRAY_SIZE-1, column)。
    // 当 w_last_o[column] 为 1 时，该列四路 result 已完成当前 K 序列的累加。
    // -------------------------------------------------------------------------
    output logic [ARRAY_SIZE-1:0]                        w_last_o,
    output logic signed [ARRAY_SIZE-1:0][ACC_WIDTH-1:0]  result_00_o,
    output logic signed [ARRAY_SIZE-1:0][ACC_WIDTH-1:0]  result_01_o,
    output logic signed [ARRAY_SIZE-1:0][ACC_WIDTH-1:0]  result_10_o,
    output logic signed [ARRAY_SIZE-1:0][ACC_WIDTH-1:0]  result_11_o
);

    // -------------------------------------------------------------------------
    // Align 后的 X 边界信号。
    //
    // 每个数组元素对应一个物理行。第 row 项已经相对原始输入延迟 row 拍，
    // 随后连接到该行最左侧的 MAC(row, 0)。
    // -------------------------------------------------------------------------
    logic                         x_valid_aligned [ARRAY_SIZE];
    logic                         x_first_aligned [ARRAY_SIZE];
    logic                         x_last_aligned  [ARRAY_SIZE];
    logic signed [DATA_WIDTH-1:0] x0_aligned      [ARRAY_SIZE];
    logic signed [DATA_WIDTH-1:0] x1_aligned      [ARRAY_SIZE];

    // Align 后的 W 边界信号。
    // 第 column 项已经相对原始输入延迟 column 拍，随后连接到该列最上方的
    // MAC(0, column)。
    logic                         w_valid_aligned [ARRAY_SIZE];
    logic                         w_first_aligned [ARRAY_SIZE];
    logic                         w_last_aligned  [ARRAY_SIZE];
    logic signed [DATA_WIDTH-1:0] w0_aligned      [ARRAY_SIZE];
    logic signed [DATA_WIDTH-1:0] w1_aligned      [ARRAY_SIZE];

    // -------------------------------------------------------------------------
    // X Align：第 row 个物理行插入 row 级寄存器。
    //
    // row=0 时不延迟，直接旁路；row>0 时，每行的数据和复制后的共享控制信号
    // 一起进入等长寄存器链。寄存器链每拍都推进，因此 valid=0 会作为气泡
    // 继续传播，不会改变后续数据之间的相对时序。
    // -------------------------------------------------------------------------
    genvar x_align_row;
    generate
        for (x_align_row = 0; x_align_row < ARRAY_SIZE; x_align_row++) begin : gen_x_align
            if (x_align_row == 0) begin : gen_x_no_delay
                assign x_valid_aligned[x_align_row] = x_valid_i;
                assign x_first_aligned[x_align_row] = x_first_i;
                assign x_last_aligned[x_align_row]  = x_last_i;
                assign x0_aligned[x_align_row]      = x0_i[x_align_row];
                assign x1_aligned[x_align_row]      = x1_i[x_align_row];
            end else begin : gen_x_delay
                // 第 x_align_row 行只定义并使用 1～x_align_row 级寄存器。
                logic                         valid_pipe [1:x_align_row];
                logic                         first_pipe [1:x_align_row];
                logic                         last_pipe  [1:x_align_row];
                logic signed [DATA_WIDTH-1:0] x0_pipe    [1:x_align_row];
                logic signed [DATA_WIDTH-1:0] x1_pipe    [1:x_align_row];

                always_ff @(posedge clk) begin
                    if (!rst_n) begin
                        for (int stage = 1; stage <= x_align_row; stage++) begin
                            valid_pipe[stage] <= 1'b0;
                            first_pipe[stage] <= 1'b0;
                            last_pipe[stage]  <= 1'b0;
                            x0_pipe[stage]    <= '0;
                            x1_pipe[stage]    <= '0;
                        end
                    end else begin
                        valid_pipe[1] <= x_valid_i;
                        first_pipe[1] <= x_first_i;
                        last_pipe[1]  <= x_last_i;
                        x0_pipe[1]    <= x0_i[x_align_row];
                        x1_pipe[1]    <= x1_i[x_align_row];

                        for (int stage = 2; stage <= x_align_row; stage++) begin
                            valid_pipe[stage] <= valid_pipe[stage-1];
                            first_pipe[stage] <= first_pipe[stage-1];
                            last_pipe[stage]  <= last_pipe[stage-1];
                            x0_pipe[stage]    <= x0_pipe[stage-1];
                            x1_pipe[stage]    <= x1_pipe[stage-1];
                        end
                    end
                end

                assign x_valid_aligned[x_align_row] = valid_pipe[x_align_row];
                assign x_first_aligned[x_align_row] = first_pipe[x_align_row];
                assign x_last_aligned[x_align_row]  = last_pipe[x_align_row];
                assign x0_aligned[x_align_row]      = x0_pipe[x_align_row];
                assign x1_aligned[x_align_row]      = x1_pipe[x_align_row];
            end
        end
    endgenerate

    // -------------------------------------------------------------------------
    // W Align：第 column 个物理列插入 column 级寄存器。共享的 W 控制信号
    // 在这里复制到各列延迟链。实现方式与 X Align 对称，使 X 与 W 在
    // MAC(row,column) 处具有相同的 row+column 总输入延迟。
    // -------------------------------------------------------------------------
    genvar w_align_column;
    generate
        for (w_align_column = 0; w_align_column < ARRAY_SIZE; w_align_column++) begin : gen_w_align
            if (w_align_column == 0) begin : gen_w_no_delay
                assign w_valid_aligned[w_align_column] = w_valid_i;
                assign w_first_aligned[w_align_column] = w_first_i;
                assign w_last_aligned[w_align_column]  = w_last_i;
                assign w0_aligned[w_align_column]      = w0_i[w_align_column];
                assign w1_aligned[w_align_column]      = w1_i[w_align_column];
            end else begin : gen_w_delay
                // 第 w_align_column 列只定义并使用 1～w_align_column 级寄存器。
                logic                         valid_pipe [1:w_align_column];
                logic                         first_pipe [1:w_align_column];
                logic                         last_pipe  [1:w_align_column];
                logic signed [DATA_WIDTH-1:0] w0_pipe    [1:w_align_column];
                logic signed [DATA_WIDTH-1:0] w1_pipe    [1:w_align_column];

                always_ff @(posedge clk) begin
                    if (!rst_n) begin
                        for (int stage = 1; stage <= w_align_column; stage++) begin
                            valid_pipe[stage] <= 1'b0;
                            first_pipe[stage] <= 1'b0;
                            last_pipe[stage]  <= 1'b0;
                            w0_pipe[stage]    <= '0;
                            w1_pipe[stage]    <= '0;
                        end
                    end else begin
                        valid_pipe[1] <= w_valid_i;
                        first_pipe[1] <= w_first_i;
                        last_pipe[1]  <= w_last_i;
                        w0_pipe[1]    <= w0_i[w_align_column];
                        w1_pipe[1]    <= w1_i[w_align_column];

                        for (int stage = 2; stage <= w_align_column; stage++) begin
                            valid_pipe[stage] <= valid_pipe[stage-1];
                            first_pipe[stage] <= first_pipe[stage-1];
                            last_pipe[stage]  <= last_pipe[stage-1];
                            w0_pipe[stage]    <= w0_pipe[stage-1];
                            w1_pipe[stage]    <= w1_pipe[stage-1];
                        end
                    end
                end

                assign w_valid_aligned[w_align_column] = valid_pipe[w_align_column];
                assign w_first_aligned[w_align_column] = first_pipe[w_align_column];
                assign w_last_aligned[w_align_column]  = last_pipe[w_align_column];
                assign w0_aligned[w_align_column]      = w0_pipe[w_align_column];
                assign w1_aligned[w_align_column]      = w1_pipe[w_align_column];
            end
        end
    endgenerate

    // -------------------------------------------------------------------------
    // MAC 阵列内部的水平 X 通路。
    //
    // 第二维长度为 ARRAY_SIZE+1：
    //   x_*_path[row][0]          是该行左边界输入；
    //   x_*_path[row][column+1]   是 MAC(row,column) 向右的输出。
    // 每经过一个 MAC，X 数据及其控制标记延迟一拍。
    // -------------------------------------------------------------------------
    logic                         x_valid_path [ARRAY_SIZE][ARRAY_SIZE+1];
    logic                         x_first_path [ARRAY_SIZE][ARRAY_SIZE+1];
    logic                         x_last_path  [ARRAY_SIZE][ARRAY_SIZE+1];
    logic signed [DATA_WIDTH-1:0] x0_path      [ARRAY_SIZE][ARRAY_SIZE+1];
    logic signed [DATA_WIDTH-1:0] x1_path      [ARRAY_SIZE][ARRAY_SIZE+1];

    // MAC 阵列内部的垂直 W 通路。
    // 第一维长度为 ARRAY_SIZE+1：第 0 行是阵列上边界，第 row+1 行是
    // MAC(row,column) 向下的输出。每经过一个 MAC 延迟一拍。
    logic                         w_valid_path [ARRAY_SIZE+1][ARRAY_SIZE];
    logic                         w_first_path [ARRAY_SIZE+1][ARRAY_SIZE];
    logic                         w_last_path  [ARRAY_SIZE+1][ARRAY_SIZE];
    logic signed [DATA_WIDTH-1:0] w0_path      [ARRAY_SIZE+1][ARRAY_SIZE];
    logic signed [DATA_WIDTH-1:0] w1_path      [ARRAY_SIZE+1][ARRAY_SIZE];

    // 保存所有物理 MAC 的四路本地累加结果。
    // 数组下标顺序统一为 [row][column]，便于后续增加行选择输出 MUX。
    logic signed [ACC_WIDTH-1:0] result_00 [ARRAY_SIZE][ARRAY_SIZE];
    logic signed [ACC_WIDTH-1:0] result_01 [ARRAY_SIZE][ARRAY_SIZE];
    logic signed [ACC_WIDTH-1:0] result_10 [ARRAY_SIZE][ARRAY_SIZE];
    logic signed [ACC_WIDTH-1:0] result_11 [ARRAY_SIZE][ARRAY_SIZE];

    // 将各行 Align 后的 X 信号连接到阵列左边界。
    genvar x_boundary_row;
    generate
        for (x_boundary_row = 0; x_boundary_row < ARRAY_SIZE; x_boundary_row++) begin : gen_x_boundary
            assign x_valid_path[x_boundary_row][0] = x_valid_aligned[x_boundary_row];
            assign x_first_path[x_boundary_row][0] = x_first_aligned[x_boundary_row];
            assign x_last_path[x_boundary_row][0]  = x_last_aligned[x_boundary_row];
            assign x0_path[x_boundary_row][0]      = x0_aligned[x_boundary_row];
            assign x1_path[x_boundary_row][0]      = x1_aligned[x_boundary_row];
        end
    endgenerate

    // 将各列 Align 后的 W 信号连接到阵列上边界。
    genvar w_boundary_column;
    generate
        for (w_boundary_column = 0; w_boundary_column < ARRAY_SIZE; w_boundary_column++) begin : gen_w_boundary
            assign w_valid_path[0][w_boundary_column] = w_valid_aligned[w_boundary_column];
            assign w_first_path[0][w_boundary_column] = w_first_aligned[w_boundary_column];
            assign w_last_path[0][w_boundary_column]  = w_last_aligned[w_boundary_column];
            assign w0_path[0][w_boundary_column]      = w0_aligned[w_boundary_column];
            assign w1_path[0][w_boundary_column]      = w1_aligned[w_boundary_column];
        end
    endgenerate

    // -------------------------------------------------------------------------
    // 生成 ARRAY_SIZE×ARRAY_SIZE 个物理 MAC 并完成二维级联。
    //
    // 对 MAC(row,column)：
    //   X 输入来自左侧 x_path[row][column]，输出到 column+1；
    //   W 输入来自上方 w_path[row][column]，输出到 row+1；
    //   四路结果保存在 result_**[row][column] 中，不在 MAC 之间传播。
    // -------------------------------------------------------------------------
    genvar mac_row;
    genvar mac_column;
    generate
        for (mac_row = 0; mac_row < ARRAY_SIZE; mac_row++) begin : gen_mac_row
            for (mac_column = 0; mac_column < ARRAY_SIZE; mac_column++) begin : gen_mac_column
                mac #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH (ACC_WIDTH)
                ) u_mac (
                    .clk        (clk),
                    .rst_n      (rst_n),

                    .x_valid_i  (x_valid_path[mac_row][mac_column]),
                    .x_first_i  (x_first_path[mac_row][mac_column]),
                    .x_last_i   (x_last_path[mac_row][mac_column]),
                    .x0_i       (x0_path[mac_row][mac_column]),
                    .x1_i       (x1_path[mac_row][mac_column]),
                    .x_valid_o  (x_valid_path[mac_row][mac_column+1]),
                    .x_first_o  (x_first_path[mac_row][mac_column+1]),
                    .x_last_o   (x_last_path[mac_row][mac_column+1]),
                    .x0_o       (x0_path[mac_row][mac_column+1]),
                    .x1_o       (x1_path[mac_row][mac_column+1]),

                    .w_valid_i  (w_valid_path[mac_row][mac_column]),
                    .w_first_i  (w_first_path[mac_row][mac_column]),
                    .w_last_i   (w_last_path[mac_row][mac_column]),
                    .w0_i       (w0_path[mac_row][mac_column]),
                    .w1_i       (w1_path[mac_row][mac_column]),
                    .w_valid_o  (w_valid_path[mac_row+1][mac_column]),
                    .w_first_o  (w_first_path[mac_row+1][mac_column]),
                    .w_last_o   (w_last_path[mac_row+1][mac_column]),
                    .w0_o       (w0_path[mac_row+1][mac_column]),
                    .w1_o       (w1_path[mac_row+1][mac_column]),

                    .result_00_o(result_00[mac_row][mac_column]),
                    .result_01_o(result_01[mac_row][mac_column]),
                    .result_10_o(result_10[mac_row][mac_column]),
                    .result_11_o(result_11[mac_row][mac_column])
                );
            end
        end
    endgenerate

    // -------------------------------------------------------------------------
    // 导出每一列最底部 MAC 的状态和结果。
    //
    // w_last_path[ARRAY_SIZE][column] 是底部 MAC 已打一拍后的 last 输出。
    // 同一上升沿内，底部 MAC 的累加寄存器已经包含最后一项乘积，因此仿真和
    // 下游逻辑可以在 w_last_o[column]==1 时直接采样对应四路 result。
    // -------------------------------------------------------------------------
    genvar output_column;
    generate
        for (output_column = 0; output_column < ARRAY_SIZE; output_column++) begin : gen_bottom_output
            assign w_last_o[output_column]    = w_last_path[ARRAY_SIZE][output_column];
            assign result_00_o[output_column] = result_00[ARRAY_SIZE-1][output_column];
            assign result_01_o[output_column] = result_01[ARRAY_SIZE-1][output_column];
            assign result_10_o[output_column] = result_10[ARRAY_SIZE-1][output_column];
            assign result_11_o[output_column] = result_11[ARRAY_SIZE-1][output_column];
        end
    endgenerate

endmodule
