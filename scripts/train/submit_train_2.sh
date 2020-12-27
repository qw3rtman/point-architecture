Executable = run_train_2.sh

+Group="GRAD"
+Project="AI_ROBOTICS"
+ProjectDescription="Training model"
+GPUJob=true

Requirements=(TARGET.GPUSlot)
Rank=memory
Universe=vanilla
Getenv=True
Notification=Complete

Log=/u/nimit/logs/$(ClusterId)_train_2.log
Output=/u/nimit/logs/$(ClusterId)_train_2.out
Error=/u/nimit/logs/$(ClusterId)_train_2.err

Queue 1
