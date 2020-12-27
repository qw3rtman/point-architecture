Executable = run_eval_14.sh

+Group="GRAD"
+Project="AI_ROBOTICS"
+ProjectDescription="Training model"
+GPUJob=true

Requirements=(TARGET.GPUSlot)
Rank=memory
Universe=vanilla
Getenv=True
Notification=Complete

Log=/u/nimit/logs/$(ClusterId)_eval_14.log
Output=/u/nimit/logs/$(ClusterId)_eval_14.out
Error=/u/nimit/logs/$(ClusterId)_eval_14.err

Queue 1
