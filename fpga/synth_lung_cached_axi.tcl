if {$argc != 1} { error "Usage: synth_lung_cached_axi.tcl NEW_OUTPUT_DIR" }
set task_root [file normalize [file join [file dirname [info script]] ..]]
set task_out [file normalize [lindex $argv 0]]
if {[string first "${task_root}/data/" $task_out] != 0} { error "Output must remain under workspace/data" }
if {[file exists $task_out]} { error "Output exists; choose a new directory" }
file mkdir $task_out
create_project -in_memory -part xc7z020clg400-1
foreach f {lung_requant.v lung_mac_tile8.v lung_mac_tile8_reuse.v lung_feature_window.v lung_conv8_window.v lung_feature_cache.v lung_axi_read64.v lung_conv8_cached_axi.v} {
    read_verilog [file join $task_root fpga $f]
}
synth_design -top lung_conv8_cached_axi -part xc7z020clg400-1 -mode out_of_context
create_clock -name core_clk -period 10.000 [get_ports clk]
report_utilization -hierarchical -file [file join $task_out utilization_hier.rpt]
report_utilization -file [file join $task_out utilization.rpt]
report_timing_summary -delay_type max -max_paths 10 -file [file join $task_out timing_synth.rpt]
write_checkpoint [file join $task_out cached_axi_synth.dcp]
puts "Feature-cache + read AXI OOC only; no write/control plane, routed timing, bit or board test."
