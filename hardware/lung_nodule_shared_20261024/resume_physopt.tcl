# Re-optimize routed placement/routing only. Never relax XDC or clock targets.
if {$argc != 1} { error "Usage: resume_lung_lut_physopt.tcl LUT_EXPERIMENT" }
set task_out [file normalize [lindex $argv 0]]
set snapshot [file normalize [file dirname [info script]]]
set task_root [file normalize [file join $snapshot ../..]]
if {[string first "${task_root}/data/" $task_out] != 0} { error "Use isolated workspace/data child" }
set backup [file join $task_out route_initial_failed]
if {[file exists $backup] || [file exists [file join $task_out deploy]]} { error "Preserve previous evidence" }
file mkdir $backup
foreach name {timing_routed.rpt utilization_routed.rpt drc.rpt configuration_exceptions.rpt} {
    file copy [file join $task_out $name] [file join $backup $name]
}
file copy [info script] [file join $task_out resume_lung_lut_physopt.tcl]
open_project [file join $task_out project lung_network.xpr]
open_run impl_1
phys_opt_design -directive AggressiveExplore
route_design -directive Explore
report_utilization -file [file join $task_out utilization_routed.rpt]
report_timing_summary -delay_type min_max -max_paths 10 -file [file join $task_out timing_routed.rpt]
report_drc -file [file join $task_out drc.rpt]
report_exceptions -file [file join $task_out configuration_exceptions.rpt]
report_route_status -file [file join $task_out route_status_physopt.rpt]
set setup [get_timing_paths -delay_type max -max_paths 1]
set hold [get_timing_paths -delay_type min -max_paths 1]
if {[llength $setup] == 0 || [get_property SLACK $setup] < 0 ||
    [llength $hold] == 0 || [get_property SLACK $hold] < 0} {
    error "Post-route physical optimization did not meet setup/hold; no bit"
}
foreach violation [get_drc_violations -quiet] {
    if {[get_property SEVERITY $violation] in {Error {Critical Warning}}} { error "DRC failed: $violation" }
}
write_checkpoint [file join $task_out routed_physopt.dcp]
file mkdir [file join $task_out deploy]
# Write the CURRENT optimized design. launch_runs would reopen the old checkpoint.
write_bitstream [file join $task_out deploy lung_network.bit]
file copy [file join $task_out project lung_network.gen sources_1 bd design_1 hw_handoff design_1.hwh] \
    [file join $task_out deploy lung_network.hwh]
puts "Unchanged 100MHz constraints: post-route physical optimization and bit generation PASS."
close_project
