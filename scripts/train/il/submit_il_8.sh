Executable = run_il_8.sh

+Group="GRAD"
+Project="AI_ROBOTICS"
+ProjectDescription="Training model"
+GPUJob=true

Requirements=(TARGET.GPUSlot)
Rank=memory
Universe=vanilla
Getenv=True
Notification=Complete

Log=/u/nimit/logs/$(ClusterId)_il_8.log
Output=/u/nimit/logs/$(ClusterId)_il_8.out
Error=/u/nimit/logs/$(ClusterId)_il_8.err

Queue 1
