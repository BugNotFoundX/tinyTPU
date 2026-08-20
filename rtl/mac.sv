`timescale 1ns/1ps


module mac #(
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32
) (
    input                           clk,
    input                           rst_n,

    // 激活，左进右出
    input                           x_valid_i,
    input                           x_first_i,
    input                           x_last_i,
    input  signed [DATA_WIDTH-1:0]  x0_i,
    input  signed [DATA_WIDTH-1:0]  x1_i,
    output                          x_valid_o,
    output                          x_first_o,
    output                          x_last_o,
    output signed [DATA_WIDTH-1:0]  x0_o,
    output signed [DATA_WIDTH-1:0]  x1_o,

    // 权重，上进下出
    input                           w_valid_i,
    input                           w_first_i,
    input                           w_last_i,
    input  signed [DATA_WIDTH-1:0]  w0_i,
    input  signed [DATA_WIDTH-1:0]  w1_i,
    output                          w_valid_o,
    output                          w_first_o,
    output                          w_last_o,
    output signed [DATA_WIDTH-1:0]  w0_o,
    output signed [DATA_WIDTH-1:0]  w1_o,

    // 结果
    output signed [ACC_WIDTH-1:0]   result_00_o, // x0 * w0
    output signed [ACC_WIDTH-1:0]   result_01_o, // x0 * w1
    output signed [ACC_WIDTH-1:0]   result_10_o, // x1 * w0
    output signed [ACC_WIDTH-1:0]   result_11_o  // x1 * w1
);

    // 1 控制寄存器
    logic                           x_valid_reg;
    logic                           x_first_reg;
    logic                           x_last_reg;
    logic signed [DATA_WIDTH-1:0]   x0_reg;
    logic signed [DATA_WIDTH-1:0]   x1_reg;

    logic                           w_valid_reg;
    logic                           w_first_reg;
    logic                           w_last_reg;
    logic signed [DATA_WIDTH-1:0]   w0_reg;
    logic signed [DATA_WIDTH-1:0]   w1_reg;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            x_valid_reg <= 1'b0;
            x_first_reg <= 1'b0;
            x_last_reg  <= 1'b0;
            x0_reg      <= '0;
            x1_reg      <= '0;

            w_valid_reg <= 1'b0;
            w_first_reg <= 1'b0;
            w_last_reg  <= 1'b0;
            w0_reg      <= '0;
            w1_reg      <= '0;
        end else begin
            x_valid_reg <= x_valid_i;
            x_first_reg <= x_first_i;
            x_last_reg  <= x_last_i;
            x0_reg      <= x0_i;
            x1_reg      <= x1_i;

            w_valid_reg <= w_valid_i;
            w_first_reg <= w_first_i;
            w_last_reg  <= w_last_i;
            w0_reg      <= w0_i;
            w1_reg      <= w1_i;
        end
    end

    // 2 乘法计算：x0 和 x1 各使用一个 dual_int8_mul，共使用两个乘号。
    // 每个子模块内部的一次宽乘法同时得到当前 x 对 w0、w1 的两个乘积。
    logic signed [15:0]             product_00_int16;
    logic signed [15:0]             product_01_int16;
    logic signed [15:0]             product_10_int16;
    logic signed [15:0]             product_11_int16;

    logic signed [ACC_WIDTH-1:0]    product_00;
    logic signed [ACC_WIDTH-1:0]    product_01;
    logic signed [ACC_WIDTH-1:0]    product_10;
    logic signed [ACC_WIDTH-1:0]    product_11;

    dual_int8_mul u_mul_x0 (
        .x_i  (x0_i),
        .w0_i (w0_i),
        .w1_i (w1_i),
        .p0_o (product_00_int16),
        .p1_o (product_01_int16)
    );

    dual_int8_mul u_mul_x1 (
        .x_i  (x1_i),
        .w0_i (w0_i),
        .w1_i (w1_i),
        .p0_o (product_10_int16),
        .p1_o (product_11_int16)
    );

    assign product_00 = {{(ACC_WIDTH-16){product_00_int16[15]}}, product_00_int16};
    assign product_01 = {{(ACC_WIDTH-16){product_01_int16[15]}}, product_01_int16};
    assign product_10 = {{(ACC_WIDTH-16){product_10_int16[15]}}, product_10_int16};
    assign product_11 = {{(ACC_WIDTH-16){product_11_int16[15]}}, product_11_int16};

    
    // 3 累加寄存器
    logic signed [ACC_WIDTH-1:0]    result_00_reg;
    logic signed [ACC_WIDTH-1:0]    result_01_reg;
    logic signed [ACC_WIDTH-1:0]    result_10_reg;
    logic signed [ACC_WIDTH-1:0]    result_11_reg;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            result_00_reg <= '0;
            result_01_reg <= '0;
            result_10_reg <= '0;
            result_11_reg <= '0;
        end
        // 数据有效时累加
        else if (x_valid_i && w_valid_i) begin
            // 如果是第一条数据，则直接赋值，否则累加
            if (x_first_i && w_first_i) begin
                result_00_reg <= product_00;
                result_01_reg <= product_01;
                result_10_reg <= product_10;
                result_11_reg <= product_11;
            end else begin
                result_00_reg <= result_00_reg + product_00;
                result_01_reg <= result_01_reg + product_01;
                result_10_reg <= result_10_reg + product_10;
                result_11_reg <= result_11_reg + product_11;
            end
        end
    end

    // 4 输出
    assign x_valid_o = x_valid_reg;
    assign x_first_o = x_first_reg;
    assign x_last_o  = x_last_reg;
    assign x0_o      = x0_reg;
    assign x1_o      = x1_reg;

    assign w_valid_o = w_valid_reg;
    assign w_first_o = w_first_reg;
    assign w_last_o  = w_last_reg;
    assign w0_o      = w0_reg;
    assign w1_o      = w1_reg;

    assign result_00_o = result_00_reg;
    assign result_01_o = result_01_reg;
    assign result_10_o = result_10_reg;
    assign result_11_o = result_11_reg;

endmodule

// =============================================================================
// 一个 DSP 同时完成两路 signed INT8 乘法。
//
// 两路乘法共享同一个激活 x：
//   p0 = x * w0
//   p1 = x * w1
//
// 将两个权重打包为 25-bit signed 数：
//   packed_weight = w0 + (w1 << 16)
//
// 再使用一次 25×18 signed 乘法：
//   packed_product = packed_weight * x
//                  = p0 + (p1 << 16)
//
// packed_product 的低 16 bit 是 p0。若 p0 为负，它的符号扩展会向高段借位，
// 使算术右移得到的高路结果等于 p1-1，因此需要在 p0<0 时加 1 修正。
// 默认位宽正好对应 Zynq-7000 DSP48E1 的 25×18 乘法器输入。
// =============================================================================
module dual_int8_mul (
    input  signed [7:0]  x_i,
    input  signed [7:0]  w0_i,
    input  signed [7:0]  w1_i,
    output signed [15:0] p0_o,
    output signed [15:0] p1_o
);

    logic signed [24:0] w0_extended;
    logic signed [24:0] w1_extended;
    logic signed [24:0] packed_weight;
    logic signed [17:0] x_extended;

    // 两路 INT8 乘积打包后只占 32 bit；DSP 的更高乘积位不参与拆包。
    (* use_dsp = "yes" *) logic signed [31:0] packed_product;

    assign w0_extended = {{17{w0_i[7]}}, w0_i};
    assign w1_extended = {{17{w1_i[7]}}, w1_i};
    assign packed_weight = w0_extended + (w1_extended <<< 16);
    assign x_extended = {{10{x_i[7]}}, x_i};

    // 本模块唯一的乘法运算，综合时应映射为一个 DSP48E1。
    assign packed_product = packed_weight * x_extended;

    assign p0_o = packed_product[15:0];
    // packed_product[31:16] 等价于取 (packed_product >>> 16) 的低 16 bit。
    assign p1_o = $signed(packed_product[31:16]) + (p0_o[15] ? 16'sd1 : 16'sd0);

endmodule
