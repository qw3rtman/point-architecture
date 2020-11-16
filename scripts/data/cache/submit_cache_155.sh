Executable = run_cache_155.sh

+Group="GRAD"
+Project="AI_ROBOTICS"
+ProjectDescription="Training model"
+GPUJob=true

Requirements=(TARGET.GPUSlot)
Rank=memory
Universe=vanilla
Getenv=True
Notification=Complete

Log=/u/nimit/logs/$(ClusterId)_cache_155.log
Output=/u/nimit/logs/$(ClusterId)_cache_155.out
Error=/u/nimit/logs/$(ClusterId)_cache_155.err

Queue 1
