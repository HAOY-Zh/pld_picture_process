if {$argc < 1 || $argc > 2} { error "Usage: synth_lung_axil.tcl NEW_OUTPUT_DIR ?UNCHANGED_RTL_CHECKPOINT?" }
set task_root [file normalize [file join [file dirname [info script]] ..]]
set task_out [file normalize [lindex $argv 0]]
if {[string first "${task_root}/data/" $task_out] != 0} { error "Output must remain under workspace/data" }
if {[file exists $task_out]} { error "Output exists; choose a new directory" }
file mkdir $task_out
if {$argc == 2} {
    set prior [file normalize [lindex $argv 1]]
    if {[string first "${task_root}/data/" $prior] != 0 || ![file exists $prior]} { error "Checkpoint must exist under workspace/data" }
    open_checkpoint $prior
    set note [open [file join $task_out checkpoint_source.txt] w]
    puts $note $prior
    close $note
    # Reuse only after confirming RTL is unchanged. The existing raw-config
    # exceptions are identical; re-read them and add derived-count exceptions.
} else {
    create_project -in_memory -part xc7z020clg400-1
    foreach f {lung_requant.v lung_mac_tile8.v lung_mac_tile8_reuse.v lung_feature_window.v lung_conv8_window.v lung_feature_cache.v lung_axi_read64.v lung_conv8_cached_axi.v lung_axi_write64.v lung_output_writer.v lung_conv8_memory_axi.v lung_axil_control.v lung_conv8_axil.v} {
        read_verilog [file join $task_root fpga $f]
    }
    synth_design -top lung_conv8_axil -part xc7z020clg400-1 -mode out_of_context
    create_clock -name core_clk -period 10.000 [get_ports clk]
}
read_xdc [file join $task_root fpga lung_axil_config_timing.xdc]
if {[llength $lung_cfg_launch] == 0 || [llength $lung_cfg_capture] == 0 || [llength $lung_cfg_derived] == 0} {
    error "Missing lung immutable-configuration endpoints; refusing silent timing exemption"
}
report_exceptions -file [file join $task_out configuration_exceptions.rpt]
report_timing -from $lung_cfg_launch -to $lung_cfg_capture -max_paths 3 -file [file join $task_out configuration_timing.rpt]
report_timing -from $lung_cfg_derived -to $lung_cfg_capture -max_paths 3 -file [file join $task_out configuration_derived_timing.rpt]
report_utilization -hierarchical -file [file join $task_out utilization_hier.rpt]
report_utilization -file [file join $task_out utilization.rpt]
report_timing_summary -delay_type max -max_paths 10 -file [file join $task_out timing_synth.rpt]
write_checkpoint [file join $task_out axil_synth.dcp]
puts "AXI-Lite plus single-block engine OOC only; no routed timing, full U-Net, bit or board proof."
