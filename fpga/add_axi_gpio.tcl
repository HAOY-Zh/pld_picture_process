# 给现有 PYNQ-Z2 Block Design 增加可编程 WL/WW 控制。
# 执行前请关闭 Vivado GUI，避免两个 Vivado 同时修改工程。

set project_file {D:/pld_picture_process/fpga/pynq_z2_window_level_axis/pynq_z2_window_level_axis.xpr}
set bd_file {D:/pld_picture_process/fpga/pynq_z2_window_level_axis/pynq_z2_window_level_axis.srcs/sources_1/bd/design_1/design_1.bd}

open_project $project_file
open_bd_design $bd_file

set ps [get_bd_cells processing_system7_0]
set control_sc [get_bd_cells smartconnect_0]
set axis [get_bd_cells window_level_axis_0]

# 如果脚本重复执行，避免重复创建 AXI GPIO。
if {[llength [get_bd_cells axi_gpio_0 -quiet]] == 0} {
    create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_0
}

# 双通道、每通道 16 bit，两个通道均为输出。
set_property -dict [list \
    CONFIG.C_IS_DUAL {1} \
    CONFIG.C_GPIO_WIDTH {16} \
    CONFIG.C_GPIO2_WIDTH {16} \
    CONFIG.C_ALL_OUTPUTS {1} \
    CONFIG.C_ALL_OUTPUTS_2 {1} \
] [get_bd_cells axi_gpio_0]

# 让控制用 SmartConnect 从一个主机扩展为两个从机：
# M00_AXI -> DMA 控制寄存器，M01_AXI -> AXI GPIO 控制寄存器。
set_property -dict [list CONFIG.NUM_MI {2}] $control_sc

# 删除旧的固定常量模块，先释放 wl_in/ww_in 两个端口。
foreach name {xlconstant_0 xlconstant_1} {
    set old_cell [get_bd_cells $name -quiet]
    if {[llength $old_cell] != 0} {
        delete_bd_objs $old_cell
    }
}

# 连接 AXI GPIO 的 AXI-Lite 控制接口。
if {[llength [get_bd_intf_nets -quiet smartconnect_0_M01_AXI]] == 0} {
    connect_bd_intf_net \
        [get_bd_intf_pins smartconnect_0/M01_AXI] \
        [get_bd_intf_pins axi_gpio_0/S_AXI]
}

# 复用现有 FCLK_CLK0 和 peripheral_aresetn；已经连接时跳过。
if {[llength [get_bd_nets -of_objects [get_bd_pins axi_gpio_0/s_axi_aclk] -quiet]] == 0} {
    connect_bd_net \
        [get_bd_pins processing_system7_0/FCLK_CLK0] \
        [get_bd_pins axi_gpio_0/s_axi_aclk]
}
if {[llength [get_bd_nets -of_objects [get_bd_pins axi_gpio_0/s_axi_aresetn] -quiet]] == 0} {
    connect_bd_net \
        [get_bd_pins proc_sys_reset_0/peripheral_aresetn] \
        [get_bd_pins axi_gpio_0/s_axi_aresetn]
}

# 清除删除常量后遗留的空网络，再连接 AXI GPIO。
foreach pin_name {wl_in ww_in} {
    set old_net [get_bd_nets -of_objects [get_bd_pins window_level_axis_0/$pin_name] -quiet]
    if {[llength $old_net] != 0} {
        delete_bd_objs $old_net
    }
}

# AXI GPIO 通道 1 输出 WL，通道 2 输出 WW。
connect_bd_net \
    [get_bd_pins axi_gpio_0/gpio_io_o] \
    [get_bd_pins window_level_axis_0/wl_in]
connect_bd_net \
    [get_bd_pins axi_gpio_0/gpio2_io_o] \
    [get_bd_pins window_level_axis_0/ww_in]

# 自动分配 AXI GPIO 的控制寄存器地址。
assign_bd_address

validate_bd_design
save_bd_design
close_bd_design [get_bd_designs design_1]
close_project
exit
