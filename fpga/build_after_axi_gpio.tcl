# 重新生成 AXI GPIO 加入后的工程输出和 bitstream。
set project_file {D:/pld_picture_process/fpga/pynq_z2_window_level_axis/pynq_z2_window_level_axis.xpr}

open_project $project_file
set bd_file [get_files -quiet */design_1.bd]
if {[llength $bd_file] == 0} {
    set bd_file [get_files -quiet *design_1.bd]
}
open_bd_design $bd_file

# 强制重建 Block Design 的所有 IP 输出产品，避免使用缺失的旧 DCP。
reset_target all $bd_file
generate_target all $bd_file

# 重新生成顶层 HDL wrapper。
set wrapper_file [make_wrapper -files $bd_file -top -force]
if {[llength [get_files -quiet $wrapper_file]] == 0} {
    add_files -norecurse $wrapper_file
}
set_property top design_1_wrapper [get_filesets sources_1]
update_compile_order -fileset sources_1

reset_run synth_1
reset_run impl_1
launch_runs synth_1 -jobs 20
wait_on_run synth_1
if {[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"} {
    error "综合失败，请检查 synth_1 日志"
}

launch_runs impl_1 -to_step write_bitstream -jobs 20
wait_on_run impl_1
if {[get_property STATUS [get_runs impl_1]] ne "write_bitstream Complete!"} {
    error "实现或生成 bitstream 失败，请检查 impl_1 日志"
}

puts "AXI GPIO bitstream build completed."
puts "BITSTREAM: [file normalize [get_property DIRECTORY [get_runs impl_1]]]/design_1_wrapper.bit"

close_project
exit
