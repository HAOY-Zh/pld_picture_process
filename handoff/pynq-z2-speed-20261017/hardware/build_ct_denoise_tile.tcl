# Build the PS/DMA/tiled denoising overlay using the local official board files.
# vivado -mode batch -source fpga/build_ct_denoise_tile.tcl -tclargs ABS_NEW_DIR BOARD_REPO ?build?
if {$argc<2 || $argc>3} {error "Usage: ABS_NEW_DIR BOARD_REPO ?build?"}
set here [file normalize [file dirname [info script]]]
set out [file normalize [lindex $argv 0]]
if {[file exists $out]} {error "Refusing to overwrite $out"}
set_param board.repoPaths [list [file normalize [lindex $argv 1]]]
set boards [get_board_parts -quiet *pynq-z2*]
if {[llength $boards]!=1} {error "Expected one PYNQ-Z2 board definition: $boards"}
create_project ct_denoise_tile $out -part xc7z020clg400-1
set_property board_part [lindex $boards 0] [current_project]
# Freeze the exact sources used by this build, keeping earlier runs auditable.
set rtl [file join $out rtl]
file mkdir $rtl
foreach name {ct_denoise_tile_axis.v ct_unsigned_divider.v} {
    file copy [file join $here $name] [file join $rtl $name]
}
file copy [file join $here ct_denoise_model ct_denoise_roms.vh] [file join $rtl ct_denoise_roms.vh]
add_files [list [file join $rtl ct_denoise_tile_axis.v] [file join $rtl ct_unsigned_divider.v] [file join $rtl ct_denoise_roms.vh]]
set_property include_dirs [list $rtl] [current_fileset]
create_bd_design design_1
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} [get_bd_cells processing_system7_0]
# Initial all-mode implementation: 50 MHz. Higher clocks require additional
# pipelining of guided-filter arithmetic and must pass routed timing again.
set clock_mhz 50.0
set clock_file [open [file join $out clock_mhz.txt] w]
puts $clock_file $clock_mhz
close $clock_file
set_property -dict [list CONFIG.PCW_USE_M_AXI_GP0 {1} CONFIG.PCW_USE_S_AXI_HP0 {1} CONFIG.PCW_FPGA_FCLK0_ENABLE {1} CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ $clock_mhz] [get_bd_cells processing_system7_0]
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0
set_property -dict [list CONFIG.c_include_sg {0} CONFIG.c_sg_length_width {23} CONFIG.c_include_mm2s_dre {0} CONFIG.c_include_s2mm_dre {0} CONFIG.c_m_axis_mm2s_tdata_width {32} CONFIG.c_m_axi_mm2s_data_width {64} CONFIG.c_m_axi_s2mm_data_width {64}] [get_bd_cells axi_dma_0]
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_0
set_property -dict [list CONFIG.C_IS_DUAL {1} CONFIG.C_GPIO_WIDTH {32} CONFIG.C_ALL_OUTPUTS {1} CONFIG.C_GPIO2_WIDTH {32} CONFIG.C_ALL_INPUTS_2 {1}] [get_bd_cells axi_gpio_0]
create_bd_cell -type module -reference ct_denoise_tile_axis ct_denoise_tile_axis_0
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 control_axi
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] [get_bd_cells control_axi]
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 memory_axi
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] [get_bd_cells memory_axi]
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 reset_0
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 locked_1
set_property -dict [list CONFIG.CONST_WIDTH {1} CONFIG.CONST_VAL {1}] [get_bd_cells locked_1]
set_property CONFIG.C_EXT_RESET_HIGH {0} [get_bd_cells reset_0]
proc link_if {a b} {connect_bd_intf_net [get_bd_intf_pins $a] [get_bd_intf_pins $b]}
link_if processing_system7_0/M_AXI_GP0 control_axi/S00_AXI
link_if control_axi/M00_AXI axi_dma_0/S_AXI_LITE
link_if control_axi/M01_AXI axi_gpio_0/S_AXI
link_if axi_dma_0/M_AXI_MM2S memory_axi/S00_AXI
link_if axi_dma_0/M_AXI_S2MM memory_axi/S01_AXI
link_if memory_axi/M00_AXI processing_system7_0/S_AXI_HP0
link_if axi_dma_0/M_AXIS_MM2S ct_denoise_tile_axis_0/s_axis
link_if ct_denoise_tile_axis_0/m_axis axi_dma_0/S_AXIS_S2MM
connect_bd_net [get_bd_pins axi_gpio_0/gpio_io_o] [get_bd_pins ct_denoise_tile_axis_0/cfg]
connect_bd_net [get_bd_pins ct_denoise_tile_axis_0/status] [get_bd_pins axi_gpio_0/gpio2_io_i]
foreach pin {processing_system7_0/M_AXI_GP0_ACLK processing_system7_0/S_AXI_HP0_ACLK axi_dma_0/s_axi_lite_aclk axi_dma_0/m_axi_mm2s_aclk axi_dma_0/m_axi_s2mm_aclk axi_gpio_0/s_axi_aclk ct_denoise_tile_axis_0/clk reset_0/slowest_sync_clk control_axi/ACLK control_axi/S00_ACLK control_axi/M00_ACLK control_axi/M01_ACLK memory_axi/ACLK memory_axi/S00_ACLK memory_axi/S01_ACLK memory_axi/M00_ACLK} {
    connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins $pin]
}
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins reset_0/ext_reset_in]
connect_bd_net [get_bd_pins locked_1/dout] [get_bd_pins reset_0/dcm_locked]
foreach pin {axi_dma_0/axi_resetn axi_gpio_0/s_axi_aresetn ct_denoise_tile_axis_0/rst_n control_axi/ARESETN control_axi/S00_ARESETN control_axi/M00_ARESETN control_axi/M01_ARESETN memory_axi/ARESETN memory_axi/S00_ARESETN memory_axi/S01_ARESETN memory_axi/M00_ARESETN} {
    connect_bd_net [get_bd_pins reset_0/peripheral_aresetn] [get_bd_pins $pin]
}
assign_bd_address
validate_bd_design
save_bd_design
set bd [get_files design_1.bd]
generate_target all $bd
set wrapper [make_wrapper -files $bd -top]
add_files -norecurse $wrapper
set_property top design_1_wrapper [current_fileset]
update_compile_order -fileset sources_1
write_bd_tcl [file join $out design_1_recreate.tcl]
if {$argc==3 && [lindex $argv 2] eq "build"} {
    launch_runs impl_1 -to_step write_bitstream -jobs 4
    wait_on_run impl_1
    if {![string match {*write_bitstream Complete*} [get_property STATUS [get_runs impl_1]]]} {error "Implementation failed"}
    open_run impl_1
    report_utilization -file [file join $out utilization_routed.rpt]
    report_timing_summary -file [file join $out timing_routed.rpt]
    report_drc -file [file join $out drc.rpt]
    set setup [get_timing_paths -quiet -delay_type max -max_paths 1]
    set hold [get_timing_paths -quiet -delay_type min -max_paths 1]
    if {[llength $setup]!=1 || [llength $hold]!=1} {error "Timing paths missing"}
    if {[get_property SLACK $setup]<0 || [get_property SLACK $hold]<0} {error "Timing failed; do not use bitstream"}
    set bit [glob -nocomplain [file join $out ct_denoise_tile.runs impl_1 *.bit]]
    set hwh [glob -nocomplain [file join $out ct_denoise_tile.srcs sources_1 bd design_1 hw_handoff *.hwh]]
    if {![llength $hwh]} {set hwh [glob -nocomplain [file join $out ct_denoise_tile.gen sources_1 bd design_1 hw_handoff *.hwh]]}
    if {[llength $bit]!=1 || [llength $hwh]!=1} {error "Matching bit/hwh pair missing"}
    file mkdir [file join $out overlay]
    file copy [lindex $bit 0] [file join $out overlay ct_denoise_tile.bit]
    file copy [lindex $hwh 0] [file join $out overlay ct_denoise_tile.hwh]
    puts "CT_TILE_OVERLAY_READY: [file join $out overlay]"
}
close_project
exit
