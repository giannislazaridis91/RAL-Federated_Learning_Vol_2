import torch.nn as nn
import torch
import numpy as np

class CentralFed():
    def __init__(self, model_example: dict, samples_per_client: list):
    	
    	# The main server.
    
        """
        Central server for the federated system

        @param model_example: Example of the State of a model to get the types of the layers and the names of layers
        @param samples_per_client: A list with the number of total images per client
        """
        self.model_example = model_example
        self.samples_per_client = samples_per_client
        self.all_samples = sum(self.samples_per_client)
        # self.weights = [torch.as_tensor(i / self.all_samples) for i in self.samples_per_client]
        self.weights = [torch.as_tensor(5) for i in self.samples_per_client]
        print(f'Length of weights for aggregation: {len(self.weights)}')
        self.agg_model = {}
        for key, value in self.model_example.items():
            self.agg_model[key] = torch.zeros_like(value, dtype=value.dtype)

    def reset_agg(self): # PRE-FINAL STEP! After the creation of the aggregated model, reset the server!
        """
        Reset the Aggregated model with it is distributed for to the local clients
        @return: Nothing
        """
        self.agg_model = {}
        for key, value in self.model_example.items():
            self.agg_model[key] = torch.zeros_like(value, dtype=value.dtype)

    def get_local_parameters(self, models: list): #Down arrow to the central node
        """
        Get the parameters of all the local clients
        @param models: List with the models of each client
        @return: A list with the parameters of each client's network
        """
        models_param_list = []
        for model in models:
            model_params = {}
            for key, param in model.named_parameters():
                model_params[key] = param

            models_param_list.append(model_params)

        return models_param_list

    def _fed_avg(self, models_param_list: list): # Aggregation of models at the server.
        """
        Perform the federated average and get the aggregated parameters from the local models

        @param models_param_list: List with the parameters from each client's network
        @return: The aggregated network
        """

        averaged_params = models_param_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(models_param_list)):
                local_model_params = models_param_list[i]
                local_sample_number = self.samples_per_client[i]
                if i == 0:
                    averaged_params[k] = (
                            local_model_params[k] * local_sample_number / self.all_samples
                    )
                else:
                    averaged_params[k] += (
                            local_model_params[k] * local_sample_number / self.all_samples
                    )
        return averaged_params

    def _update_local_models(self, aggregated_model: dict, local_models: list): # Return a list of the aggregated model, one for each client.
        """
        Update the local clients with the aggregated model

        @param aggregated_model: The parameters of the aggregated model from the local clients
        @param local_models: The list of the local models that the weights will be updated to
        @return: Nothing special
        """
        with torch.no_grad():
            for model in local_models:
                for key, value in model.named_parameters():
                    value.copy_(aggregated_model[key])

        return local_models

    def fed_avg(self, models: list, ):

        assert len(models) == len(self.weights), 'Number of client models and split of the dataset is not the same!'

        for key, value in models[0].items():

            type_of_weights = value.dtype
            break

        for i in range(len(models)):
            model = models[i]
            weight = self.weights[i].type(dtype=type_of_weights)
            for key, value in model.items():
                if 'bn' in key or value.dtype == torch.int64:
                    value_np = value.numpy()
                    value_np = value_np.astype(np.float32)
                    weight_np = weight.numpy()
                    value_to_add = value_np * weight_np
                    value_to_add = np.round(value_to_add)
                    value_to_add = value_to_add.astype(np.int64)
                    value_to_add = np.array(value_to_add)
                    value_to_add = torch.from_numpy(value_to_add)
                    self.agg_model[key] += value_to_add
                else:
                    self.agg_model[key] += value * weight

        return self.agg_model



"""

Here I have an example of how we can update the weights of the local models with the aggregated model

og_models = [copy.deepcopy(model) for model in models] # List of models.
models_params = central_server.get_local_parameters(models) # List of dictionaries of state dictionaries of the models.
aggregated_model = central_server._fed_avg(models_params) # Aggregated model's creation.
updated_model = central_server._update_local_models(aggregated_model, models) # The model, one for each client.
new_models = [copy.deepcopy

for client in range(num_clients):
    dif = compute_layerwise_difference(og_models[client], models[client])
    plot_layerwise_difference(dif, f'round_{round}_client_{client}')

"""
