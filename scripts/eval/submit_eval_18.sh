Executable = run_eval_18.sh

+Group="GRAD"
+Project="AI_ROBOTICS"
+ProjectDescription="Training model"
+GPUJob=true

Requirements=(TARGET.GPUSlot)
Rank=memory
Universe=vanilla
Getenv=True
Notification=Complete

Log=/u/nimit/logs/$(ClusterId)_eval_18.log
Output=/u/nimit/logs/$(ClusterId)_eval_18.out
Error=/u/nimit/logs/$(ClusterId)_eval_18.err

Queue 1
