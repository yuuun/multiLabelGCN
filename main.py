import math
import random
from torch.nn import MultiLabelSoftMarginLoss
from sklearn.metrics import classification_report
from data_loader import Data
import torch.optim as optim
from models import *
import torch
import time

if __name__=='__main__':
    n_class = 6
    six_field = ['Machine learning',
                'Distributed computing',
                'Real-time computing',
                'Data mining',
                'Computer network',
                'Pattern recognition']

    data = Data('../dataset/6field/', n_class)
    
    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)
    
        
    criterion = MultiLabelSoftMarginLoss()
    
    model = GCN(nfeat = data.features.shape[1], nhid=160, nclass=n_class, dropout=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    lab = torch.LongTensor(data.label_list)

    
    for epoch in range(0, 10001):
        start = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(data.features, data.nfadj)
        loss_train = criterion(output[data.idx_train], lab[data.idx_train])
            
        loss_train.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimzier_state_dict': optimizer.state_dict(),
                'loss': loss_train,
                }, './noSigmoid.pt')
            print('{:2d} {:.4f} {:.2f}s\n'.format(epoch, loss_train.item(), time.time() - start), end='\t')

            with torch.no_grad():
                model.eval()
                output = model(data.features, data.nfadj)
                
                outputs = nn.Sigmoid()(output) > 0.5  # uses sigmoid function in test case instead of train case
                kk = classification_report(outputs[data.idx_test].tolist(), lab[data.idx_test].tolist())
                print(kk)
