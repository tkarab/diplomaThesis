import tensorflow as tf
from tensorflow import keras
import os
import json

class IterationLoggingCallback(keras.callbacks.Callback):
    # def on_batch_end(self, batch, logs=None):
    #     if (batch % 100) == 0:
    #         # print(f"Batch {batch + 1}: loss = {logs.get('loss'):.2f}\n")
    #         print()
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        # print('win_size: ', win_size)

class TrainingInfoCallback(keras.callbacks.Callback):
    def __init__(self, file_path, model,  batch_size, model_filename, model_backbone_name, experiment, iterations_per_epoch, validation_steps, preprocessing_dict, aug_enabled, aug_dict, data_intake, rms, db,best_epoch_kept = 0):
        super(TrainingInfoCallback, self).__init__()
        self.file_path = file_path
        self.model = model
        self.batch_size = batch_size
        self.best_epoch_lr = float(self.model.optimizer.learning_rate.numpy())

        # 'model_protoNet_1.h5' -> 'model_protoNet_1'
        model_filename = model_filename.split('.')[0]
        self.json_filename = model_filename + "_training_info.json"
        self.txt_res_filename = model_filename + "_results.txt"
        self.json_fullpath = os.path.join(self.file_path, self.json_filename)
        self.txt_res_fullpath = os.path.join(self.file_path, self.txt_res_filename)

        self.model_backbone_name = model_backbone_name
        self.experiment = experiment
        self.iterations_per_epoch = iterations_per_epoch
        self.validation_steps = validation_steps
        self.preprocessing_dict = preprocessing_dict
        self.aug_enabled = aug_enabled
        self.aug_dict = aug_dict
        self.best_epoch_kept = best_epoch_kept

        self.data_intake = data_intake
        self.rms        = rms
        self.db         = db

    def round_results(self,logs):
        return {key: round(value, 4) for key, value in logs.items()}

    def update_json_log_file(self,epoch):
        if os.path.exists(self.json_fullpath):
            with open(os.path.join(self.file_path,self.json_filename), 'r') as f:
                training_info = json.load(f)
            # training_info["RESULTS"][f"epoch {epoch+1}"] = self.round_results(logs)
            training_info["TRAINING_INFO"]["TOTAL_EPOCHS"] = epoch+1
            training_info["TRAINING_INFO"]["BEST_EPOCH_KEPT"] = self.best_epoch_kept
            training_info["TRAINING_INFO"]["LATEST_EPOCH_LEARNING_RATE"] = float(self.model.optimizer.learning_rate.numpy())
            training_info["TRAINING_INFO"]["BEST_EPOCH_LEARNING_RATE"] = self.best_epoch_lr

        else:
            # Create a dictionary with training info
            training_info = {
                "MODEL" : {
                    "NAME" : self.model.name,
                    "BASE" : self.model_backbone_name
                },
                "PROCESSING" : {
                    "PREPROCESSING" : self.preprocessing_dict,
                    "AUG_ENABLED"   : self.aug_enabled,
                    "AUGMENTATION"  : self.aug_dict
                },
                "TRAINING_INFO" : {
                    "EXPERIMENT" : self.experiment,
                    "BATCH_SIZE" : self.batch_size,
                    "ITERATIONS_PER_EPOCH" : self.iterations_per_epoch,
                    "VALIDATION_STEPS" : self.validation_steps,
                    "OPTIMIZER" : self.model.optimizer._name,
                    "LOSS" : self.model.loss,
                    "METRICS" : self.model.metrics_names,
                    "TOTAL_EPOCHS" : 1,
                    "LATEST_EPOCH_LEARNING_RATE": float(self.model.optimizer.learning_rate.numpy()),
                    "BEST_EPOCH_KEPT" : self.best_epoch_kept,
                    "BEST_EPOCH_LEARNING_RATE": float(self.model.optimizer.learning_rate.numpy())
                },
                "DATA_GENERATOR" : {
                    "DATA_INTAKE" : self.data_intake,
                    "RMS" : self.rms,
                    "DB" : self.db
                }
            }
        # Save the dictionary as a JSON file
        with open(os.path.join(self.file_path,self.json_filename), 'w') as file:
            json.dump(training_info, file, indent=4)

    def write_txt_res_line(self,epoch,logs):
        train_acc = logs['train_accuracy']
        train_loss = logs['train_loss']
        val_acc = logs['val_accuracy']
        val_loss = logs['val_loss']

        # Format the result line
        result_line = f"{epoch + 1:<5}\t{train_acc:<15.4f}\t{train_loss:<15.4f}\t{val_acc:<15.4f}\t{val_loss:<15.4f}\n"

        # Append the result line to the file
        with open(self.txt_res_fullpath, 'a') as f:
            f.write(result_line)
    def update_txt_results_file(self,epoch,logs=None):
        if not os.path.exists(self.txt_res_fullpath):
            with open(self.txt_res_fullpath,'w') as res_file:
                column_names = f"{'Epoch':<5}\t{'Train Accuracy':<15}\t{'Train Loss':<15}\t{'Val Accuracy':<15}\t{'Val Loss':<15}\n"
                res_file.write(column_names)
                res_file.write("-" * 65 + "\n")

        self.write_txt_res_line(epoch,logs)

    # logs has the following form: {'train_loss':1.2, 'train_accuracy':0.6, 'val_loss':1.4, 'val_accuracy':0.5}
    def on_epoch_end(self, epoch, logs=None):
        self.update_json_log_file(epoch)
        self.update_txt_results_file(epoch,logs)

        return
