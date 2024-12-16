
class TaskExecutor:
    def __init__(self, model_data):
        self.model_data = model_data
    def execute_model(self):

        mobile_flops = self.model_data['mobile_flops']
        server_flops = self.model_data['server_flops']
        datasize = self.model_data['size']

        return mobile_flops, server_flops, datasize

