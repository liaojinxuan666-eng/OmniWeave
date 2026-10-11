class MetaWeaveController:
    def __init__(self, model, uncertainty_estimator):
        self.model = model
        self.unc = uncertainty_estimator
        
    def generate(self, input_ids, max_retries=2):
        for i in range(max_retries):
            outputs = self.model(input_ids)
            # 评估不确定性
            score = self.unc(outputs)
            if score < 0.8: # 置信度够高
                return outputs
            else:
                # 触发重试，或者在这里接入工具/大模型
                print(f"不确定性高 ({score})，触发重试...")
                # 你可以在这里修改输入，或者调用外部API
        return outputs