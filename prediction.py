import pickle
import torch


class PredictImportance(torch.nn.Module):
    def __init__(self, n_vocab):
        super().__init__()
        self.embed = torch.nn.Embedding(n_vocab, 100)
        self.rnn = torch.nn.RNN(100, 256)
        self.linear = torch.nn.Linear(256, 1)

    def forward(self, inputs):
        embeded = self.embed(inputs)
        embeded = torch.transpose(embeded, 0, 1)
        output, hidden = self.rnn(embeded)
        logits = self.linear(hidden.squeeze(0))
        return logits


def execute_prediction(word_to_id, model, string):
    token = [word_to_id[w] for w in string.strip().split()]

    model.eval()
    with torch.no_grad():
        inputs = torch.tensor([token]).to(torch.device("cpu"))
        logits = model(inputs).squeeze(1)
        rounded_sigmoidedlogits = torch.round(torch.sigmoid(logits))
    result = "imp" if rounded_sigmoidedlogits == 1 else "oth"  # "imp" means important, "oth" means normal
    return result


with open('word_to_id.pkl', 'rb') as f:
    word_to_id = pickle.load(f)

with open('id_to_word.pkl', 'rb') as f:
    id_to_word = pickle.load(f)


with open('forpred_all_raw_inputs.pkl', 'rb') as f:
    forpred_all_raw_inputs = pickle.load(f)

# make model to execute prediction
model = PredictImportance(len(word_to_id))
model.to(torch.device("cpu"))

save_dict = torch.load("importance_result.pth")
model.load_state_dict(save_dict['state_dict'])

# saving final results to pickle for deploy.py
final_result = []
for i in forpred_all_raw_inputs:
    final_result.append(execute_prediction(word_to_id, model, i))


with open('final_result.pkl', 'wb') as f:
    pickle.dump(final_result, f)

print("done\n")
