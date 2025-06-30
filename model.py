import torch
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Model
from dataset.py import SequenceDataset

class T5(nn.Module):
  def __init__(self, tokenizer,columns=['page','event','category','subcategory'],model_name='t5-small'):
    super().__init__()
    self.columns=columns
    self.tokenizer=tokenizer
    self.vocab=tokenizer.vocab
    self.vocab_size = tokenizer.get_vocab_size()
    # Load pretrained T5
    self.t5 = T5Model.from_pretrained(model_name,output_hidden_states=True)

    for col in self.columns:
      setattr(self, f"{col}_head", nn.Linear(self.t5.config.d_model, self.vocab_size))
  def apply_vocab_mask(self,logits, mask):
    mask = mask.to(logits.device).view(1, 1, -1)
    return logits.masked_fill(~mask, -1e9)


  def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,page_labels=None,event_labels=None,category_labels=None,subcategory_labels=None):
    # Use T5 encoder-decoder
    output = self.t5(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        return_dict=True
    )
    hidden_states =output.decoder_hidden_states[-1] # [B, L, d_model]
    outputs={}
    for col in self.columns:
        head = getattr(self, f"{col}_head")
        logits =head(hidden_states)
        logits =self.apply_vocab_mask(logits, self.tokenizer.mask(col))
        outputs[f"logits_{col}"] =logits

    self.outputs = outputs
    return self.outputs
  
  def compute_loss(self,logits_event,logits_page,**label_kwargs):
    loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.vocab["<pad>"])
    total_loss = 0.0
    losses = {}
    for col in self.columns:
      logits = self.outputs[f"logits_{col}"]
      labels = label_kwargs[f"{col}_labels"]

      loss = loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))
      losses[f"loss_{col}"] = loss
      total_loss += loss

    # store in outputs
    self.outputs['loss'] = total_loss
    self.outputs.update(losses)
    return self.outputs
    
  def train_model(self,
    csv_train,csv_val,                 
    num_epochs=5,
    batch_size=8,
    lr=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu"):
    

    # Dataset and DataLoader
    train_dataset = SequenceDataset(csv_train, self.tokenizer, text_column="X", chunksize=10)
    Val_dataset = SequenceDataset(csv_val, self.tokenizer, text_column="X", chunksize=10)
    loader_train = DataLoader(train_dataset, batch_size=batch_size)
    loader_val = DataLoader(Val_dataset, batch_size=batch_size)

    optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    # Training loop
    self.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        progress = tqdm(loader_train, desc=f"Epoch {epoch+1}")
        batch_idx=0

        for batch in progress:
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # === Forward pass with teacher forcing ===
            outputs = self(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=batch["decoder_input_ids"]
            )

            # === Compute loss ===
            losses = self.compute_loss(
                logits_page=outputs["logits_page"],
                logits_event=outputs["logits_event"],
                page_labels=batch["page_labels"],
                event_labels=batch["event_labels"],
                category_labels= batch["category_labels"],
                subcategory_labels=batch["subcategory_labels"]
            )

            loss = losses["loss"]
            total_loss += loss.item()
            batch_idx+=1


            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update progress bar
            progress.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / batch_idx
        print(f"\nEpoch {epoch+1} avg loss: {avg_loss:.4f}")

    print("Training complete.")
    
    self.eval()
    with torch.no_grad():
        val_loss, val_batches = 0, 0
        for batch in tqdm(loader_val, desc=f"Epoch {epoch+1} [Val]"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = self(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=batch["decoder_input_ids"]
            )

            losses = self.compute_loss(
                logits_page=outputs["logits_page"],
                logits_event=outputs["logits_event"],
                page_labels=batch["page_labels"],
                event_labels=batch["event_labels"],
                category_labels= batch["category_labels"],
                subcategory_labels=batch["subcategory_labels"]
            )

            val_loss += losses["loss"].item()
            val_batches += 1

        avg_val_loss = val_loss / val_batches
        print(f"📕 Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
    self.train()
  def beam_search(self, sequence, beam_width=3, max_len=5, device="cpu"):
    self.eval()
    with torch.no_grad():
        batch = self.prep_batch(sequence, self.tokenizer)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Start with decoder input: <pad>
        start_token_id = self.tokenizer.vocab["<pad>"]
        decoder_input_ids = torch.full((1, 1), start_token_id, dtype=torch.long).to(device)

        # Beam = list of (log_prob, decoder_input_ids, page_seq, event_seq)
        beam = [(0.0, decoder_input_ids, [], [])]

        for step in range(max_len):
            new_beam = []

            for log_prob, dec_input, page_seq, event_seq in beam:
                outputs = self(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=dec_input
                )

                # Alternate prediction heads based on step
                if step % 2 == 0:  # Page
                    logits = outputs["logits_page"][:, -1, :]
                else:             # Event
                    logits = outputs["logits_event"][:, -1, :]

                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)  # [vocab_size]

                # Top-k tokens
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)

                for i in range(beam_width):
                    token_id = topk_indices[i].item()
                    token_log_prob = topk_log_probs[i].item()

                    new_input = torch.cat(
                        [dec_input, torch.tensor([[token_id]], device=device)], dim=1
                    )

                    new_page_seq = page_seq + [token_id] if step % 2 == 0 else page_seq
                    new_event_seq = event_seq + [token_id] if step % 2 == 1 else event_seq

                    new_beam.append((log_prob + token_log_prob, new_input, new_page_seq, new_event_seq))

            # Keep top-k beams
            beam = sorted(new_beam, key=lambda x: x[0], reverse=True)[:beam_width]

        # Final best sequence
        best_log_prob, best_input, best_page_ids, best_event_ids = beam[0]
        id2tok = self.tokenizer.idx2token
        page_tokens = [id2tok[i] for i in best_page_ids if i != self.tokenizer.vocab["<pad>"]]
        event_tokens = [id2tok[i] for i in best_event_ids if i != self.tokenizer.vocab["<pad>"]]

        return page_tokens, event_tokens




