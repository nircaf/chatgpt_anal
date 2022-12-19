from dsp.src.data_preparation import *
from dsp.src.EEGDatasets import *

from dsp.src.Models_scripts.from_1D_to_3D_pipeline_NN import *
import dsp.src.Models_scripts.Clustering as clustering
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from dsp.src.Visualization.Plots_after_training import *

from dsp.src.Create_Datasets import *
from dsp.src.Visualization.Visual_my_networks import *
import time
from dsp.src.Models_scripts.Clustering_UnSupervised_utils import *
import torchvision.transforms as transforms
from sklearn.metrics.cluster import normalized_mutual_info_score
from torch.distributions.multivariate_normal import MultivariateNormal
import math


plt.close('all')

def make_soft_true_label(True_label=None, args=None):
    uniform_exp_factor = (args.nmb_cluster) / (args.n_classes)
    output = torch.zeros((len(True_label), args.nmb_cluster))
    if args.type_of_true_label_distribution == 'gaussian':
        x = torch.linspace(-0.5, 0.5, int(uniform_exp_factor))
        pdf = (1 / (args.var_true_label * np.sqrt(2 * math.pi))) * torch.exp(-0.5 * (x / args.var_true_label)**2)
        for b_inx in range(len(True_label)):
            if True_label[b_inx] == 0:
                shift = uniform_exp_factor // 2
            elif True_label[b_inx] == (args.n_classes-1):
                shift = -uniform_exp_factor // 2
            else:
                shift = 0
            output[b_inx, int(int(True_label[b_inx] * uniform_exp_factor) - (uniform_exp_factor // 2) + shift) : int(int(True_label[b_inx] * uniform_exp_factor) + (uniform_exp_factor // 2) + shift)] = pdf



        return output.cuda()


def train(loader, model, crit_unsupervised, crit_supervised, opt, epoch, args=None):
    """Training of the CNN.
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): CNN
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()

    # switch to train mode
    model.train()

    # create an optimizer for the last fc layer
    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=args.lr_rate,
        weight_decay=args.weight_decay,
    )

    end = time.time()
    for i, (input_tensor, psaudo_target, true_target) in enumerate(loader):
        data_time.update(time.time() - end)

        # save checkpoint
        n = len(loader) * epoch + i
        if n % args.checkpoints == 0:
            path = os.path.join(
                args.exp,
                'checkpoints',
                'checkpoint_' + str(n / args.checkpoints) + '.pth.tar',
            )
            if args.verbose:
                print('Save checkpoint at: {0}'.format(path))
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : opt.state_dict()
            }, path)

        target = psaudo_target.cuda()
        input_var = torch.autograd.Variable(input_tensor.cuda())
        target_var = torch.autograd.Variable(target)

        output = model(input_var.type(torch.cuda.FloatTensor))

        true_lbls = make_soft_true_label(True_label=true_target, args=args)
        output_var = torch.autograd.Variable(output)
        exm_output = (torch.matmul(output_var, (true_lbls + torch.tensor([1e-5], requires_grad=True).cuda()))).detach()
        exm_output.requires_grad_()
        loss = crit_unsupervised(exm_output, target_var)
        # loss_supervised = crit_supervised(true_lbls, target_var)

        # loss = (1e-3) * loss_unsupervised + loss_supervised
        # record loss
        losses.update(loss.item(), input_tensor.size(0))

        # compute gradient and do SGD step
        opt.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        opt.step()
        optimizer_tl.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 200) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(loader), batch_time=batch_time,
                          data_time=data_time, loss=losses))


    return losses.avg

def compute_features(dataloader, model, N, args=None):
    if args.verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor, _) in enumerate(dataloader):
        input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
        aux = model(input_var.type(torch.cuda.FloatTensor)).data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * args.batch_size: (i + 1) * args.batch_size] = aux
        else:
            # special treatment for final batch
            features[i * args.batch_size:] = aux

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 200) == 0:
            print('{0} / {1}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  .format(i, len(dataloader), batch_time=batch_time))
    return features


def main_unsupervised_clustering(args, model, dataset, labels_name=None):
    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # CNN
    # if args.verbose:
    #     print('Architecture: {}'.format(args.arch))
    fd = int(model.top_layer.weight.size()[1])
    model.top_layer = None
    # model.features = torch.nn.DataParallel(model.features)
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    # create optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr_rate,
        momentum=args.momentum,
        weight_decay=10**args.weight_decay,
    )

    # define loss function
    criterion_unsupervised = nn.CrossEntropyLoss().cuda()
    criterion_supervised = nn.CrossEntropyLoss().cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # remove top_layer parameters from checkpoint
            for key in checkpoint['state_dict']:
                if 'top_layer' in key:
                    del checkpoint['state_dict'][key]
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # creating checkpoint repo
    exp_check = os.path.join(args.exp, 'checkpoints')
    if not os.path.isdir(exp_check):
        os.makedirs(exp_check)

    # creating cluster assignments log
    cluster_log = Logger(os.path.join(args.exp, 'clusters'))

    # preprocessing of data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tra = [transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           normalize]

    # load the data
    end = time.time()
    if args.verbose:
        print('Load dataset: {0:.2f} s'.format(time.time() - end))

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             pin_memory=True)

    # clustering algorithm to use
    deepcluster = clustering.__dict__[args.clustering](args.nmb_cluster)

    best_nmi = 0
    nmi = 0
    # training convnet with DeepCluster
    for epoch in range(args.start_epoch, args.n_epoch):
        end = time.time()

        # remove head
        model.top_layer = None
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

        # get the features for the whole dataset
        features = compute_features(dataloader, model, len(dataset), args=args)

        # cluster the features
        if args.verbose:
            print('Cluster the features')
        clustering_loss = deepcluster.cluster(features, verbose=args.verbose)

        # assign pseudo-labels
        if args.verbose:
            print('Assign pseudo labels')
        item = dataset.x
        true_target = dataset.y
        if labels_name is None:
            train_dataset = clustering.cluster_assign(deepcluster.images_lists,
                                                      item=item, True_target=true_target, args=args, epoch=epoch)
        else:
            train_dataset = clustering.cluster_assign(deepcluster.images_lists,
                                                      item=item, True_target=true_target, args=args, epoch=epoch, labels_name=labels_name)

        # uniformly sample per target
        sampler = UnifLabelSampler(int(args.reassign * len(train_dataset)),
                                   deepcluster.images_lists)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            sampler=sampler,
            pin_memory=True,
        )

        # set last fully connected layer
        mlp = list(model.classifier.children())
        mlp.append(nn.ReLU(inplace=True).cuda())
        model.classifier = nn.Sequential(*mlp)
        model.top_layer = nn.Linear(fd, len(deepcluster.images_lists))
        model.top_layer.weight.data.normal_(0, 0.01)
        model.top_layer.bias.data.zero_()
        model.top_layer.cuda()

        # train network with clusters as pseudo-labels
        end = time.time()
        loss = train(train_dataloader, model, crit_unsupervised=criterion_unsupervised, crit_supervised=criterion_supervised, opt=optimizer, epoch=epoch, args=args)

        # print log
        if args.verbose:
            print('###### Epoch [{0}] ###### \n'
                  'Time: {1:.3f} s\n'
                  'Clustering loss: {2:.3f} \n'
                  'ConvNet loss: {3:.3f}'
                  .format(epoch, time.time() - end, clustering_loss, loss))
            try:
                nmi = normalized_mutual_info_score(
                    clustering.arrange_clustering(deepcluster.images_lists),
                    clustering.arrange_clustering(cluster_log.data[-1])
                )
                print('NMI against previous assignment: {0:.3f}'.format(nmi))
            except IndexError:
                pass
            print('####################### \n')
        # save running checkpoint

        if nmi > best_nmi:
            best_nmi = nmi
            torch.save({'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict()},
                       os.path.join(args.exp, 'checkpoint.pth.tar'))

        # save cluster assignments
        cluster_log.log(deepcluster.images_lists)



if __name__ == '__main__':
    args = None
