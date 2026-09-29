# Code Task Resources

该目录保存代码修复环境使用的最小仓库资源和独立测试逻辑。`qdp_case.py` 提供一个可复现的失败用例、源文件内容与隔离测试执行。

它不是 Agent runtime；代码工具只通过 `CodeRepairEnv` 暴露，最终成功仍由 verifier 独立重跑测试并检查修改范围。
