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
    # TODO - take into account changes regarding best_val_loss etc
    def __init__(self, file_path, model,  batch_size, model_filename, model_backbone_name, experiment, iterations_per_epoch, validation_steps, preprocessing_dict, aug_enabled, aug_dict, data_intake, rms, db):
        super(TrainingInfoCallback, self).__init__()
        self.file_path = file_path
        self.model = model
        self.batch_size = batch_size

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


        self.best_epoch_val_loss = 0
        self.best_epoch_val_acc = 0

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

        self.best_loss_lr = self.get_lr()
        self.best_acc_lr = self.get_lr()

        self.load_results_if_exist()

        self.data_intake = data_intake
        self.rms         = rms
        self.db          = db

    def load_results_if_exist(self):
        if os.path.exists(self.json_fullpath):
            if os.path.exists(self.json_fullpath):
                with open(self.json_fullpath, 'r') as f:
                    training_info = json.load(f)

            self.best_epoch_val_loss = training_info["RESULTS"]["BEST_EPOCH_LOSS"]
            self.best_epoch_val_acc = training_info["RESULTS"]["BEST_EPOCH_ACC"]
            self.best_val_loss = training_info["RESULTS"]["BEST_VAL_LOSS"]
            self.best_val_acc = training_info["RESULTS"]["BEST_VAL_ACC"]
            self.best_loss_lr = training_info["RESULTS"]["BEST_EPOCH_LOSS_LEARNING_RATE"]
            self.best_acc_lr = training_info["RESULTS"]["BEST_EPOCH_ACC_LEARNING_RATE"]


    def round_results(self,logs):
        return {key: round(value, 4) for key, value in logs.items()}

    def get_lr(self):
        return float(self.model.optimizer.learning_rate.numpy())
    def update_json_log_file(self,epoch):
        if os.path.exists(self.json_fullpath):
            with open(self.json_fullpath, 'r') as f:
                training_info = json.load(f)
            # training_info["RESULTS"][f"epoch {epoch+1}"] = self.round_results(logs)
            training_info["RESULTS"]["TOTAL_EPOCHS"] = epoch+1
            training_info["RESULTS"]["LATEST_EPOCH_LEARNING_RATE"] = self.get_lr()

            training_info["RESULTS"]["BEST_VAL_LOSS"] = self.best_val_loss
            training_info["RESULTS"]["BEST_EPOCH_LOSS"] = self.best_epoch_val_loss
            training_info["RESULTS"]["BEST_EPOCH_LOSS_LEARNING_RATE"] = self.best_loss_lr

            training_info["RESULTS"]["BEST_VAL_ACC"] = self.best_val_acc
            training_info["RESULTS"]["BEST_EPOCH_ACC"] = self.best_epoch_val_acc
            training_info["RESULTS"]["BEST_EPOCH_ACC_LEARNING_RATE"] = self.best_acc_lr

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
                },
                "RESULTS": {
                    "TOTAL_EPOCHS": epoch + 1,
                    "BEST_VAL_LOSS": self.best_val_loss,
                    "BEST_VAL_ACC": self.best_val_acc,
                    "BEST_EPOCH_LOSS": self.best_epoch_val_loss,
                    "BEST_EPOCH_ACC": self.best_epoch_val_acc,
                    "LATEST_EPOCH_LEARNING_RATE": self.get_lr(),
                    "BEST_EPOCH_LOSS_LEARNING_RATE": self.best_loss_lr,
                    "BEST_EPOCH_ACC_LEARNING_RATE": self.best_acc_lr
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

class ReduceLrOnPlateauCustom(keras.callbacks.Callback):
    def __init__(self, model, reduction_factor, min_delta, best_val_loss, patience, cooldown_patience, epochs_without_improvement=0, cooldown_counter=0, min_lr=1e-5):
        super(ReduceLrOnPlateauCustom,self).__init__()

        # Model and values
        self.model = model
        self.min_delta = min_delta
        self.best_val_loss = best_val_loss
        self.min_lr = min_lr
        self.reduction_factor = reduction_factor

        # Lr reduction patience
        self.patience = patience
        self.epochs_without_improvement = epochs_without_improvement

        # Cooldown patience
        self.cooldown_patience = cooldown_patience
        self.cooldown_counter = cooldown_counter

    def on_epoch_end(self, epoch, logs=None):
        current_lr = float(self.model.optimizer.learning_rate.numpy())
        val_loss = logs['val_loss']
        val_accuracy = logs['val_accuracy']
        min_lr_reached = (current_lr <= self.min_lr)
        if min_lr_reached == True:
            return True
        improvement_made = False

        # Improvement
        if val_loss < self.best_val_loss - self.min_delta:
            print(f"new best loss {val_loss:.4f}")
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            improvement_made = True

        # If in cooldown mode no need to make any change yet
        if self.cooldown_counter > 0:
            print("You are in cooldown mode")
            self.cooldown_counter -= 1
            return min_lr_reached

        # No improvement
        if not improvement_made:
            self.epochs_without_improvement += 1
            # If Maximum epochs without improvement reached
            if self.epochs_without_improvement >= self.patience:
                # Calculate new lr
                # multiply current lr with the reduction factor
                current_lr = current_lr * self.reduction_factor

                if current_lr <= self.min_lr:
                    min_lr_reached = True
                    new_lr = self.min_lr
                    print(f"Minimum learning rate of {self.min_lr} reached")
                else:
                    new_lr = current_lr

                print(f"Reducing learning rate to {new_lr}")
                self.model.optimizer.learning_rate.assign(new_lr)

                # Reset epochs without improvement counter and enter cooldown
                self.epochs_without_improvement = 0
                print("Entering cooldown")
                self.cooldown_counter = self.cooldown_patience

        return min_lr_reached


class ReduceLrSteadilyCustom(keras.callbacks.Callback):
    def __init__(self,model,reduction_factor,patience, patience_counter=0, min_lr=1e-5):
        super(ReduceLrSteadilyCustom,self).__init__()
        self.model = model
        self.reduction_factor = reduction_factor
        self.patience = patience
        self.counter = patience_counter
        self.min_lr = min_lr
        
    def on_epoch_end(self, epoch, logs=None):
        self.counter += 1
        current_lr = float(self.model.optimizer.learning_rate.numpy())
        min_lr_reached = False

        if current_lr <= self.min_lr:
            print("Minimum lr already reached")
            return True
        # If reached enough epochs
        if self.counter >= self.patience:
            current_lr = current_lr*self.reduction_factor

            if current_lr <= self.min_lr:
                print(f"Minimum lr {self.min_lr:.6} reached")
                new_lr = self.min_lr
                min_lr_reached = True
            else:
                new_lr = current_lr

            print(f"New lr value: {new_lr:.6f}")
            self.model.optimizer.learning_rate.assign(new_lr)
            self.counter = 0

        return min_lr_reached

