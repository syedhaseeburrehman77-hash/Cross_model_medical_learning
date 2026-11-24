from abc import ABC, abstractmethod
import torch
import copy
class BaseTrainer(ABC):
    """Common interface every algorithm trainer must implement."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def compute_loss(self, model, outputs, labels, criterion) -> torch.Tensor:
        """Return the (possibly augmented) loss for a single batch."""
        raise NotImplementedError

    def on_train_start(self, model):
        """Hook called once before local training begins.
        Override if the algorithm needs to snapshot state (e.g. global params)."""
        pass

    def train(self, model, trainloader, epochs, lr, device, valloader=None, patience=3):
        """Shared training loop — same for every algorithm.
        Only compute_loss / on_train_start differ per strategy."""
        self.on_train_start(model)

        model.to(device)
       
        criterion = torch.nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            epoch_batches = 0
            
            for batch in trainloader:
                images, labels = batch["img"].to(device), batch["label"].to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = self.compute_loss(model, outputs, labels, criterion)
                loss.backward()
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        raise RuntimeError("NaN/Inf gradients detected")
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                epoch_loss += loss.item()      # Accumulates loss for this epoch only
                epoch_batches += 1             # Counts batches in this epoch
                
            # Early stopping based on validation loss
            if valloader is not None:
                model.eval()
                val_loss = 0.0
                val_batches = 0

                with torch.no_grad():
                    for batch in valloader:
                        images = batch["img"].to(device)
                        labels = batch["label"].to(device)

                        outputs = model(images)
                        loss = self.compute_loss(
                            model, outputs, labels, criterion
                        )

                        val_loss += loss.item()
                        val_batches += 1

                if val_batches > 0:
                    val_loss /= val_batches
                    print(
                        f"Epoch {epoch + 1}/{epochs}, "
                        f"train_loss={epoch_loss / epoch_batches:.4f}, "
                        f"val_loss={val_loss:.4f}"
                    )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = copy.deepcopy(
                            model.state_dict()
                        )
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        print(
                            f"Early stopping at epoch {epoch + 1}. "
                            f"Best val_loss={best_val_loss:.4f}"
                        )
                        break

        # Restore weights from the best validation epoch
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return total_loss / num_batches if num_batches > 0 else 0.0