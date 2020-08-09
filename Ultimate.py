import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchsample as ts
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pytorchtools
from pytorchtools import EarlyStopping
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print("We're using =>", device)
config = {
        'lr': 1e-3,
        'momentum': 0.9,
        'weight_decay': 0.001,
        'batch_size': 8,
        'device': 'cuda:0',
        'seed': 42,
        'input_size' : 178,
        'hidden_size_1': 1,
        'hidden_size_2': 1,
        'num_classes' : 2, 
        'num_epochs' : 1,
        'dropout' : 0.25
        }

batch_size = 8
shuffle_dataset = True
seed = 50
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
#torch.backends.cudnn.deterministic=True


class MapDataset(torch.utils.data.Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index):
        if self.map:     
            x = self.map(self.dataset[index][0]) 
        else:     
            x = self.dataset[index][0]  # image
        y = self.dataset[index][1]   # label      
        return x, y

    def __len__(self):
        return len(self.dataset)

class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        # The sequential container object in PyTorch is designed to make it
        # simple to build up a neural network layer by layer.

        self.feat_extractor = nn.Sequential(
            
            #nn.Conv2d(3, 64, 3, padding = 1, stride = 1),  # Applies a 2D convolution over an input signal
            nn.Conv2d(3, 32, 3, stride = 1),
            # composed of several input planes.
            # We use the Conv2d layer because our image data is two dimensional.
            # nn.Conv2d(1, 64, 3) with
            # in_channels=1 as we use greyscale,
            # out_channels=64,
            # kernel_size=3 is the size of the filter that is run over the images.

            nn.ReLU(),  # Applies the rectified linear unit function element-wise:

            nn.BatchNorm2d(32),  # Applies Batch Normalization over a
            # 4D input (a mini-batch of 2D inputs with additional
            # channel dimension) as described in the paper WITH num_features = 64

            nn.MaxPool2d(2,2),  # Applies a 2D max pooling over an input signal
            # composed of several input planes.

            #nn.Conv2d(64, 128, 3, padding = 1, stride = 1),  # Applies a 2D convolution over an input signal
            nn.Conv2d(32, 64, 3, stride = 1),
            # composed of several input planes.

            nn.ReLU(),  # Applies the rectified linear unit function element-wise:

            nn.BatchNorm2d(64),  # Applies Batch Normalization over a
            # 4D input (a mini-batch of 2D inputs with additional
            # channel dimension) as described in the paper WITH num_features = 128
            
            
            nn.MaxPool2d(2),  # Applies a 2D max pooling over an input signal
            # composed of several input planes.
            
            nn.Conv2d(64, 128, 3, stride = 1),
            # composed of several input planes.

            nn.ReLU(),  # Applies the rectified linear unit function element-wise:

            nn.BatchNorm2d(128),  # Applies Batch Normalization over a
            # 4D input (a mini-batch of 2D inputs with additional
            # channel dimension) as described in the paper WITH num_features = 128
            
            
            nn.MaxPool2d(2)  # Applies a 2D max pooling over an input signal
            # composed of several input planes.
            
            
        )

        self.classifier = nn.Sequential(
            
            nn.Dropout(0.2),

            nn.Linear(128 * 196  , 1024),  # Applies a linear transformation to the
            # incoming data WITH in_features = 25 * 128, out_features = 1024

            nn.ReLU(),  # Applies the rectified linear unit function element-wise:
                
            
            nn.Dropout(0.2),    

            nn.Linear(1024, 512),  # Applies a linear transformation to the
            # incoming data WITH in_features = 1024, out_features = 1024

            nn.ReLU(),  # Applies the rectified linear unit function element-wise:

            
            #nn.Linear(1024, 2)  # Applies a linear transformation to the
            # incoming data WITH in_features = 1024, out_features = 10
            
            nn.Dropout(0.5),    

            nn.Linear(512, 256),  # Applies a linear transformation to the
            # incoming data WITH in_features = 1024, out_features = 1024

            nn.ReLU(),  # Applies the rectified linear unit function element-wise:

            nn.Linear(256, 1)  # Applies a linear transformation to the
            # incoming data WITH in_features = 1024, out_features = 10

            
        )

    def forward(self, x):
        # print("forward")
        #print("feat_extractor")
        x = self.feat_extractor(x)
        n, c, h, w = x.shape
        x = x.view(n, -1)
        #print("classifier")
        x = self.classifier(x)
        return x

def acc(y_pred, y_test):
    y_pred_tag = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_tag, dim = 1)
    correct_results_sum = (y_pred_tags == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    #acc = torch.round(acc * 100)
    return acc

def binary_acc_2(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    #acc = torch.round(acc * 100)
    
    return acc



if __name__ == '__main__':

    

    transform = transforms.Compose([
        #transforms.Resize(130),
        #transforms.CenterCrop(128),
        transforms.RandomResizedCrop(128,scale=(0.2,1)),
        #transforms.Resize((80,120)),
        #transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(0.5),
        #torchvision.transforms.ColorJitter(saturation=0.05, hue=0.05),
        #torchvision.transforms.RandomRotation((-30,30)),
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])
    
    transform2 = transforms.Compose([
        transforms.Resize(130),
        transforms.CenterCrop(128),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(0.2),
        #torchvision.transforms.ColorJitter(saturation=0.05, hue=0.05),
        #torchvision.transforms.RandomRotation((-45,45)),
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])
    
    
    # Whole dataset that must be splited
    full_dataset = torchvision.datasets.ImageFolder("C:/Users/chris/Documents/2020_Università/AI/Project/final_train")
    #other_test_dataset = torchvision.datasets.ImageFolder("C:/Users/chris/Documents/2020_Università/AI/Project/Test", transform = transform2)
    
    #test_dataset = torchvision.datasets.ImageFolder("C:/Users/chris/Documents/2020_Università/AI/Project/predict",transform = transform)
    # Creating PT data samplers and loaders:

    train_val_size = int(0.85 * len(full_dataset))
    test_size = len(full_dataset) - train_val_size
    train_val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_val_size, test_size])
    
    train_size = int(0.83 * train_val_size)
    val_size = train_val_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, val_size])
    
    
    num_images = len(train_dataset)
    
    train_dataset = MapDataset( train_dataset , transform)
    val_dataset = MapDataset(val_dataset, transform2)
    test_dataset = MapDataset(test_dataset, transform2)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=False, num_workers=6, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size = 128, shuffle=False, num_workers=6, drop_last=True)
    #test_loader = DataLoader(dataset=other_test_dataset, batch_size = 1, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size = 1, shuffle=True)
    
    
    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated
    
    '''
    # Get a batch of training data
    train_loader_iter = iter(train_loader)
    inputs, classes = next(train_loader_iter)
    
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    class_names = full_dataset.classes
    imshow(out, title=[class_names[x] for x in classes])
    '''

    
    # mapping of ID to class.
    #idx2class = {v: k for k, v in train_dataset.dataset.class_to_idx.items()}


    
    #plt.imshow(single_image.(1, 2, 0))
    #print(config['input_size'])
    model = CNN()
    #model.load_state_dict(torch.load('wow.pt'))
    
    #print(model)
    model.to(device)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    #print(optimizer)
    patience = 30
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
    early_stopping = EarlyStopping(patience= patience, verbose=True, delta= 0.0025)
    accuracy_stats = {'train': [],'val': []}
    #loss_stats = {'train': [1.12, 0.37204634404757564, 0.267335427238753, 0.24446988236485867, 0.2250530268800886, 0.20706896996942528, 0.1990761392864219, 0.1895652343437337, 0.18093369652827582, 0.17491882196382472, 0.16862745991532216, 0.15923180806924375, 0.1567417831535925, 0.15393750743711726, 0.14735408641986156, 0.14653498574829937, 0.13886180739000178, 0.13757579332511677, 0.13790141425111838, 0.1365066236375194, 0.1330885149603873, 0.12780454315310508, 0.1264026062236282, 0.1223788325462425, 0.1252436771367987, 0.12344388348426212, 0.12056647981271933, 0.11798700775232232, 0.11543851365384303, 0.1161723952380973, 0.11491507697000838, 0.11135763763205002, 0.10881626138692363, 0.10857360922780476, 0.1077346183326945, 0.10676528760150336, 0.10425161307252813, 0.10373898783469931, 0.10372708355517764, 0.10658309878291268, 0.09845500191052754, 0.10004538624433049, 0.09760293482165587, 0.0987628682990346, 0.09651129141351894, 0.09803374003815024, 0.09744670086033773, 0.0923737218244034, 0.09186921541562729, 0.09222924286140162, 0.08820943602998006, 0.08882666904511943, 0.09568479635932467, 0.09409239427431633, 0.08706117405866583, 0.08940125283736147, 0.0891524995922258, 0.09397881241155821, 0.08019907158603401, 0.08306634482077993, 0.08724353534349225, 0.08258624966338016, 0.08213072181515918, 0.08031472624197863, 0.09060131007674754, 0.09303734189291533, 0.0848206457469547, 0.08463416664387312, 0.07936659543529938, 0.07773403575839966, 0.08000268110711324, 0.07258795611887124, 0.08787909380094916, 0.07759322234217012, 0.07813161266804264, 0.07484039959595784, 0.07999641405777973, 0.0757518002902272, 0.07676160085145711, 0.07888186235683398, 0.08685340323044281, 0.07403182772625434, 0.07192374691939014, 0.0700327782593504, 0.07260755740776005],
#                  'val': [0.4, 0.2573264005283515, 0.23553948352734247, 0.22234716080129147, 0.18869454972445965, 0.18308807350695133, 0.19814710194865862, 0.1756627233698964, 0.14905669338380298, 0.17665772636731467, 0.1492146390179793, 0.13209518518609306, 0.17813793756067753, 0.12978842792411646, 0.13761231411869326, 0.13404959828282395, 0.1535283986789485, 0.15150689457853636, 0.12763281119987369, 0.12353947743152578, 0.15802169234181443, 0.12141765366929273, 0.11747790190080802, 0.16610286151990294, 0.10620822051229577, 0.11679047377159198, 0.10658364075546463, 0.10743416822515428, 0.12359699017057817, 0.1162236959207803, 0.10481124557554722, 0.1550084037395815, 0.11310490670924385, 0.12608693811732033, 0.10384068350928526, 0.1189775753300637, 0.106360065129896, 0.10556140006519854, 0.11457488973004122, 0.11385780021858712, 0.10574759915471077, 0.09547621803358197, 0.10663385270163417, 0.11127240971351664, 0.10798548231832683, 0.1082266557496041, 0.15231007523834705, 0.10671926926200588, 0.09633827946769695, 0.10297936806455255, 0.10895259662841757, 0.09868847283845146, 0.10571778216399252, 0.10418707776504259, 0.09131474304012954, 0.10446259767437975, 0.11163939232937992, 0.10383087718704094, 0.09933811736603577, 0.10364526968138914, 0.09402110582838456, 0.09873740615633626, 0.09760295681189746, 0.09283306705765426, 0.10650188468086223, 0.11237765033729374, 0.10388277595241864, 0.10629147500731051, 0.11333015584386885, 0.0969027536145101, 0.0961070506212612, 0.10120371511826913, 0.10880030148352186, 0.09169317793566734, 0.09245063061825931, 0.12147740189296503, 0.14178627698371807, 0.15967327821999788, 0.12645260609375933, 0.1011755174646775, 0.13304969482123852, 0.09441989574891825, 0.09929232206195593, 0.10641990827086072, 0.1168409608847772]}
    loss_stats = {'train': [],'val': []}
    #minposs = 55
    
    y_pred_list = []
    y_true = [] 
    eP = 300
    
    min_valid_loss = 1000
    import time
    start_time = time.time()
    
    minposs = 0
    
    for epoch in range(eP):
        print( "Epoch: ", epoch,"/",eP)
    
        #TRAIN ###############################################################
        model.train()
        train_epoch_loss = 0
        acc_train = 0
        train_epoch_acc = 0
        y_true = []
        y_pred = []
        for data, target in train_loader:
    
            #LOADING THE DATA IN A BATCH
            #data, target = i
            # moving the tensors to the configured device
            data, target = data.to(device), target.to(device)
            
            
            #FORWARD PASS
            #output = model(data).squeeze()
            #print(target.unsqueeze(1).size())
            target_for_binary = target.unsqueeze(1)
            output = model(data)#.squeeze()
            #print(output.size())
            #output = model(data)
            #output = torch.unsqueeze(output, 0)
            #print(output.data)
            #print(target.data)
            target_for_binary = target_for_binary.type_as(output)
            #target = target.type(torch.FloatTensor)
            loss = criterion(output, target_for_binary)
            # calculating the total_loss for checking
            train_epoch_loss += loss.item() ###LOSS
    
            #BACKWARD AND OPTIMIZE
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # PREDICTIONS
    
            #pred = np.round(output)
            #target = target.float()
            #y_true.extend(target.tolist())
            #y_pred.extend(pred.reshape(-1).tolist())
    
            acc_train = binary_acc_2( output , target_for_binary)
            train_epoch_acc += acc_train.item() #TRAIN
    
    
        #VALIDATION ###############################################################
        #model in eval mode skips Dropout etc
        model.eval()
        val_epoch_loss = 0
        val_epoch_acc = 0
        y_true = []
        y_pred = []
        # set the requires_grad flag to false as we are in the test mode
        with torch.no_grad():
            for data, target in val_loader:
    
                #LOAD THE DATA IN A BATCH
                #data,target = i
    
                # moving the tensors to the configured device
                data, target = data.to(device), target.to(device)
    
                # Create model on data
                output = model(data)#.squeeze()
                #output = torch.unsqueeze(output, 0)
                target_for_binary = target.unsqueeze(1)
                target_for_binary = target_for_binary.type_as(output)
                #target = target.type(torch.FloatTensor)
                loss = criterion(output, target_for_binary)
                # calculating the total_loss for checking
                val_epoch_loss += loss.item()
    
                #PREDICTIONS
    
                #pred = np.round(output)
                #target = target.float()
                #y_true.extend(target.tolist())
                #y_pred.extend(pred.reshape(-1).tolist())
    
                acc_val = binary_acc_2( output , target_for_binary)
                val_epoch_acc += acc_val.item()
    
        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
    
        print( "Epoch: ",epoch,"/",eP)
        print( "Train Loss: ", (train_epoch_loss/len(train_loader)))
        print( "Val Loss: ", (val_epoch_loss/len(val_loader)))
        print( "Train Acc: ", (train_epoch_acc/len(train_loader)))
        print( "Val Acc: ", (val_epoch_acc/len(val_loader)))
        
        if (val_epoch_loss/len(val_loader)) <= min_valid_loss:
            print('Validation loss decreased ({} --> {}).  Saving model ...'.format(min_valid_loss,(val_epoch_loss/len(val_loader))))
            torch.save(model.state_dict(), 'model.pt')
            min_valid_loss = (val_epoch_loss/len(val_loader))
        
        
        if ((epoch % 50) == 0 ):
                train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
                train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30,10))
                sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
                sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')
                plt.show()
                # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        
        #print(loss_stats['val'][epoch])
        early_stopping(loss_stats['val'][epoch ], model)
        
        if early_stopping.early_stop:
            minposs = epoch +1 - patience
            print("Early stopping")
            break
    
    
    train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30,10))
    sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
    sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')
    plt.show()
    #print("________________________________________________________________________")
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(loss_stats['val'])+1),loss_stats['train'], label='Training Loss')
    plt.plot(range(1,len(loss_stats['val'])+1),loss_stats['val'],label='Validation Loss')
    
    # find position of lowest validation loss
     
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
    
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.5) # consistent scale
    plt.xlim(0, len(loss_stats['val'])+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('loss_plot.png', bbox_inches='tight')
    
    
    model.eval()
    y_true = []
    y_pred = []
    accuracyVAL = 0
    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_test_pred = model(x_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            #y_test_pred = torch.log_softmax(y_test_pred, dim=1)
            #_, y_pred_tag = torch.max(y_test_pred, dim = 1)
            y_pred.append(y_pred_tag.cpu().numpy())
            y_true.append(y_batch.cpu().numpy())
    
        
        '''
        testAcc = binary_acc(y_pred,y_true)
        print ("Accuracy on test set is" , testAcc)
        accuracyVAL = testAcc
        print ("***********************************************************")
        '''
        
        import itertools
        y_pred_list = list(itertools.chain.from_iterable(y_pred))
        y_true_list = list(itertools.chain.from_iterable(y_true))
        print(classification_report(y_true_list, y_pred_list))
        testAcc = accuracy_score(y_true_list,y_pred_list)
        print ("Accuracy on test set is 1" , testAcc)
        accuracyVAL = testAcc
        print ("***********************************************************")

    
    print ("________________________________________________________________________")
    
    
    layers = list(model.state_dict())
    
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_true_list, y_pred_list))
    #print "Dropout fixed at: ", dropout
    print (confusion_matrix_df)
    confusion_matrix_df = confusion_matrix_df.rename(columns={0 : "Anime", 1 : "Cartoon", 2 : "Comic", 3 : "Manga"  }, index={0 : "Anime", 1 : "Cartoon" , 2 : "Comic", 3 : "Manga" })
    #confusion_matrix_df = confusion_matrix_df.rename(columns={0 : "Anime", 1 : "Cartoon", 2 : "Comic", 3 : "Manga" }, index={0 : "Anime", 1 : "Cartoon", 2 : "Comic", 3 : "Manga" })
    print (confusion_matrix_df)
    # print model
    
    ax = plt.axes()
    sns.heatmap(confusion_matrix_df, annot=True,fmt="d")
    name = 'Total Test Set:' , len(test_dataset)
    ax.set_title('Confusion Matrix')
    #print ("1 (Seizure ) on TestSet: ", b[1], " 0 (Not Seizure) on TestSet: ", b[0])
    # CONFUSION MATRIX
    #print  (classification_report(y_true, y_pred, target_names=["Not Seizure", "Seizure"]))
    plt.show()
    
    
    
    model.load_state_dict(torch.load('checkpoint.pt'))
    torch.save(model.state_dict(), 'earlystopped.pt')
    model.eval()
    y_true = []
    y_pred = []
    accuracyVAL = 0
    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_test_pred = model(x_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            #y_test_pred = torch.log_softmax(y_test_pred, dim=1)
            #_, y_pred_tag = torch.max(y_test_pred, dim = 1)
            y_pred.append(y_pred_tag.cpu().numpy())
            y_true.append(y_batch.cpu().numpy())
    
        '''
        testAcc = binary_acc(y_pred,y_true)
        print ("Accuracy on test set is" , testAcc)
        accuracyVAL = testAcc
        print ("***********************************************************")
        '''
        import itertools
        y_pred_list = list(itertools.chain.from_iterable(y_pred))
        y_true_list = list(itertools.chain.from_iterable(y_true))
        print(classification_report(y_true_list, y_pred_list))
        testAcc = accuracy_score(y_true_list,y_pred_list)
        print ("Accuracy on test set is 2" , testAcc)
        accuracyVAL = testAcc
        print ("***********************************************************")

    
    print ("________________________________________________________________________")
    
    
    layers = list(model.state_dict())
    
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_true_list, y_pred_list))
    #print "Dropout fixed at: ", dropout
    print (confusion_matrix_df)
    confusion_matrix_df = confusion_matrix_df.rename(columns={0 : "Anime", 1 : "Cartoon", 2 : "Comic", 3 : "Manga"  }, index={0 : "Anime", 1 : "Cartoon" , 2 : "Comic", 3 : "Manga" })
    #confusion_matrix_df = confusion_matrix_df.rename(columns={0 : "Anime", 1 : "Cartoon", 2 : "Comic", 3 : "Manga" }, index={0 : "Anime", 1 : "Cartoon", 2 : "Comic", 3 : "Manga" })
    print (confusion_matrix_df)
    # print model
    
    ax = plt.axes()
    sns.heatmap(confusion_matrix_df, annot=True,fmt="d")
    name = 'Total Test Set:' , len(test_dataset)
    ax.set_title('Confusion Matrix')
    #print ("1 (Seizure ) on TestSet: ", b[1], " 0 (Not Seizure) on TestSet: ", b[0])
    # CONFUSION MATRIX
    #print  (classification_report(y_true, y_pred, target_names=["Not Seizure", "Seizure"]))
    plt.show()
    print("--- %s seconds ---" % (time.time() - start_time))