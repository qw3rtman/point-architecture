Executable = run_cache_90.sh

+Group="GRAD"
+Project="AI_ROBOTICS"
+ProjectDescription="Training model"
+GPUJob=true

Requirements=(TARGET.GPUSlot)
Rank=memory
Universe=vanilla
Getenv=True
Notification=Complete

Log=/u/nimit/logs/$(ClusterId)_cache_90.log
Output=/u/nimit/logs/$(ClusterId)_cache_90.out
Error=/u/nimit/logs/$(ClusterId)_cache_90.err

Queue 1
