import torch
from tqdm import tqdm
from datetime import datetime
import os
import copy
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, 
                 model, 
                 train_dataloader, 
                 val_dataloader, 
                 epochs, 
                 criterion, 
                 optimizer, 
                 save_path: str = "./checkpoints",
                 model_name: str = "model.pth"):
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = save_path
        self.model_name = model_name
        self.model.to(self.device)
        os.makedirs(self.save_path, exist_ok=True)

        self.train_losses = []
        self.val_losses = []

    def train(self, patience: int = 10):
        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(self.model.state_dict())
        epochs_no_improve = 0

        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            train_loss = self.train_one_epoch()
            val_loss = self.evaluate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            tqdm.write(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                epochs_no_improve = 0
                tqdm.write(f"\nValidation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                self.save_model(best_model_wts, epoch, best_val_loss)
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                tqdm.write(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
                break
        
        self.model.load_state_dict(best_model_wts)

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        for images, masks in tqdm(self.train_dataloader, desc="Training", leave=False):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, masks)
        loss.backward()
        self.optimizer.step()

        running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(self.train_dataloader.dataset)
        return epoch_loss

    def evaluate(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(self.val_dataloader, desc="Evaluating", leave=False):
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(self.val_dataloader.dataset)
        return epoch_loss
    
    def save_model(self, model_state, epoch, metric: float):
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"epoch_{epoch+1}_metric_{metric:.4f}_{self.model_name}"
        save_filepath = os.path.join(self.save_path, date_str, filename)
        os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
        torch.save(model_state, save_filepath)
        print(f"Model saved to {save_filepath}")
    
    def plot_losses(self):
        date_str = datetime.now().strftime("%Y%m%d")
        lowest_val_loss = min(self.val_losses)
        figure_name = f"loss_curve_lowest_val_{lowest_val_loss:.4f}.png"
        os.makedirs(os.path.join(self.save_path, date_str), exist_ok=True)

        plt.figure(figsize=(10, 5))
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Loss Curve - Lowest Val Loss: {lowest_val_loss:.4f}')
        plt.show()
        plt.savefig(os.path.join(self.save_path, date_str, figure_name))