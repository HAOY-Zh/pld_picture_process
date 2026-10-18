if {$argc != 1} { error "Usage: synth_lung_tensor.tcl NEW_OUTPUT_DIR" }
set task_root [file normalize [file join [file dirname [info script]] ..]]
set task_out [file normalize [lindex $argv 0]]
if {[string first "${task_root}/data/" $task_out] != 0 || [file exists $task_out]} { error "Need a fresh directory under data" }
file mkdir $task_out
create_project -in_memory -part xc7z020clg400-1
foreach f {lung_requant.v lung_axi_read64.v lung_axi_write64.v lung_tensor_engine.v lung_tensor_control.v lung_tensor_axil.v} {
    read_verilog [file join $task_root fpga $f]
}
synth_design -top lung_tensor_axil -part xc7z020clg400-1 -mode out_of_context
create_clock -name core_clk -period 10.000 [get_ports clk]
report_utilization -file [file join $task_out utilization.rpt]
report_timing_summary -delay_type min_max -file [file join $task_out timing.rpt]
report_timing -max_paths 8 -file [file join $task_out critical_paths.rpt]
write_checkpoint [file join $task_out tensor_synth.dcp]
set setup [get_timing_paths -delay_type max -max_paths 1]
if {[llength $setup] == 0 || [get_property SLACK $setup] < 0} { error "Tensor internal setup timing not met; do not build board" }
puts "Tensor independent synthesis and internal setup check passed; not routed/board proof."
close_project
