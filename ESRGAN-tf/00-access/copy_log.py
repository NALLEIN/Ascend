import moxing as mox
mox.file.copy_parallel("/var/log/npu/slog", "s3://esrgan-tf/host_log")
