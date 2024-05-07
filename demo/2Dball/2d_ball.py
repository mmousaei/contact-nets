import pdb  # noqa

import click
import matplotlib.pyplot as plt
import numpy as np
import torch

# plt.switch_backend('pgf')

def generate_figure(regenerate_data):
    '''
    TRAINING PARAMETERS
    '''
    global xf, xf_plot, models, legend, v0_plot, v0, PENNLGRAY2, PENNYELLOW, LINEWIDTH, colors
    # dataset sizes
    N = 20  # training & test dataset size
    N_batch = 1  # batch size
    N_batches = int(N / N_batch)  # number of batches

    # network params
    H = 128  # hidden units

    # optimizer parameters
    learning_rate = 1e-4
    N_epoch = 10000  # max epochs
    estop = 50  # early stopping patience in epochs



    '''
    PHYSICS/SYSTEM PARAMETERS
    '''
    # gravity, timestep, true ground height, true friction coeficient
    g, dt, GH, Mu = 9.81, 0.1, 0, 0.1
    # initial 2D position
    q0 = torch.tensor([1.0, 0.0]).unsqueeze(0)  # Shape [1, 2]

    # predictor of final state (q'v')
    def nominalModel(v0, gh, mu, restitution=0.8):
        if v0.dim() == 1:
            v0 = v0.unsqueeze(0)
        qF_z = q0[:, 0] + v0[:, 0] * dt - 0.5 * g * dt**2
        qF_y = q0[:, 1] + v0[:, 1] * dt

        vF_z = v0[:, 0] - g * dt
        vF_y = v0[:, 1]

        contact_mask = (qF_z < gh)
        vF_z[contact_mask] = -restitution * vF_z[contact_mask]  # Apply restitution
        vF_y[contact_mask] *= (1 - mu)  # Apply friction

        qF_z = torch.max(qF_z, torch.tensor(gh))  # Ensure the position does not go below ground height

        qF = torch.stack([qF_z, qF_y], dim=1)
        vF = torch.stack([vF_z, vF_y], dim=1)
        return torch.cat((qF, vF), dim=1)



    '''
    MODEL CONSTRUCTION
    '''
    # END-TO-END MODEL
    Nonlinearity = torch.nn.Tanh
    # Nonlinearity = torch.nn.ReLU
    modelF = torch.nn.Sequential(
        torch.nn.Linear(2, H),
        Nonlinearity(),
        torch.nn.Linear(H, H),
        Nonlinearity(),
        torch.nn.Linear(H, H),
        Nonlinearity(),
        torch.nn.Linear(H, 4),
    )

    def weighted_loss(output, target, weight_position=10.0, weight_velocity=1.0):
        loss_z_position = ((output[:, 0] - target[:, 0]) ** 2).mean()
        loss_y_position = ((output[:, 1] - target[:, 1]) ** 2).mean()
        loss_z_velocity = ((output[:, 2] - target[:, 2]) ** 2).mean()
        loss_y_velocity = ((output[:, 3] - target[:, 3]) ** 2).mean()

        # Summing up with weights applied
        total_loss = (weight_position * (loss_z_position + 10*loss_y_position) +
                    weight_velocity * (loss_z_velocity + 5*loss_y_velocity))
        
        return total_loss

    # lossF = torch.nn.MSELoss(reduction='mean')
    lossF = weighted_loss

    # ContactNets Model

    class GroundHeightQP(torch.nn.Module):
        """
        GROUND-HEIGHT MODEL WITH QP LOSS that learns both the ground height and friction coefficient.
        """
        def __init__(self, gh_initial, mu_initial):
            super().__init__()
            # Ground height parameter
            self.gh = torch.nn.Parameter(torch.tensor([gh_initial], dtype=torch.float32), requires_grad=True)
            # Friction coeficient parameter
            self.mu = torch.nn.Parameter(torch.tensor([mu_initial], dtype=torch.float32), requires_grad=True)

        def forward(self, data):
            gh = self.gh.clone().detach()
            gh.requires_grad_(True)
            mu = self.mu.clone().detach()
            mu.requires_grad_(True)
            return nominalModel(data[:, :2], gh, mu)
        
    modelGH = GroundHeightQP(GH, Mu)
    MSELOSS = torch.nn.MSELoss(reduction='mean')
    RELU = torch.nn.ReLU()

    def lossGH(predictions, true_final, gh, mu):
        """
        Compute loss for the GroundHeightQP model considering contact dynamics.
        
        Args:
        predictions (Tensor): The output from the model containing predicted states.
        true_final (Tensor): The actual final states from the data.
        gh (float): Ground height parameter.
        mu (float): Friction coefficient parameter.

        Returns:
        Tensor: The computed loss value.
        """
        pred_qf = predictions[:, :2]  # Predicted final positions [qf_z, qf_y]
        pred_vf = predictions[:, 2:]  # Predicted final velocities [vf_z, vf_y]

        # Prediction Quality Loss
        pred_quality_loss_q = MSELOSS(pred_qf, true_final[:, :2])
        pred_quality_loss_v = MSELOSS(pred_vf, true_final[:, 2:])

        # Contact Activation Loss
        # Activate only if predicted final vertical position is at or below ground level
        contact_mask = pred_qf[:, 0] <= gh
        contact_activation_loss = MSELOSS(torch.zeros_like(pred_qf[:, 0]), RELU(pred_qf[:, 0] - gh)) * contact_mask.float()

        # Non-penetration Loss
        non_penetration_loss = MSELOSS(torch.zeros_like(pred_qf[:, 0]), RELU(gh - pred_qf[:, 0]))

        # Maximal Dissipation Loss
        # Using predicted friction to calculate the dissipation in horizontal velocity
        friction_loss = MSELOSS(torch.zeros_like(pred_vf[:, 1]), (1 - mu) * (true_final[:, 1] - pred_vf[:, 1])) * contact_mask.float()

        # Combining the losses
        total_loss = pred_quality_loss_q + pred_quality_loss_v + contact_activation_loss + non_penetration_loss + friction_loss

        return total_loss

    # learning parameters
    models = [modelF, modelGH]
    # models = [modelGH]
    # models = [modelF]
    losses = [lossF, lossGH]
    # losses = [lossGH]
    # losses = [lossF]
    testlosses = [lossF] * len(models)
    catter = lambda x_, y_: torch.cat((x_, y_), dim=1)  # noqa
    mdata = [(lambda x_, y_: x_)] * len(models)
    ldata = [(lambda x_, y_: y_)] * (len(models)) 
    mdata_t = [(lambda x_, y_: x_)] * len(models)
    ldata_t = [(lambda x_, y_: y_)] * len(models)
    # early_stop_epochs = [120, 50, 120, 120, 120, 120]

    legend = ['Baseline (end2end)', 'ContactNets']
    savef = ['modelF_2D.pt', 'modelGH_2D.pt']
    rates = [learning_rate] * len(models)
    opts = [torch.optim.Adam(m.parameters(), lr=learning_rate) for m in models] * len(models)


    # early stopping parameters
    early_stop_epochs = [estop] * len(models)
    best_loss = [10000.] * len(models)
    best_epoch = [0] * len(models)
    end = [False]* len(models)


    '''
    DATA SYNTHESIS
    '''
    # set range of initial velocities
    # center initial velocity v0 around impact
    v_center_z = g / 2 * dt - q0[:, 0].squeeze() / dt
    SC = .5
    v0min_z = v_center_z * (1 - SC)
    v0max_z = v_center_z * (1 + SC)
    v0min_y = -0.5
    v0max_y =  0.5
    STD = 0.1  # noise standard deviation

    if regenerate_data:
        # generate training data
        v0_z = (v0max_z - v0min_z) * torch.rand(N, 1) + v0min_z
        v0_y = (v0max_y - v0min_y) * torch.rand(N, 1) + v0min_y
        
        v0 = torch.cat((v0_z.T,v0_y.T)).T

        xf = nominalModel(v0, GH, Mu)

        # generate test data
        v0_t_z = (v0max_z - v0min_z) * torch.rand(N, 1) + v0min_z
        v0_t_y = (v0max_y - v0min_y) * torch.rand(N, 1) + v0min_y
        v0_t = torch.cat((v0_t_z.T,v0_t_y.T)).T
        xf_t = nominalModel(v0_t, GH, Mu)

        # generate plotting data
        v0_plot_z = torch.linspace(v0_z.min(), v0_z.max(), N * 100).unsqueeze(1)
        v0_plot_y = torch.linspace(v0_y.min(), v0_y.max(), N * 100).unsqueeze(1)
        v0_plot = torch.cat((v0_plot_z.T, v0_plot_y.T)).T

        xf_plot = nominalModel(v0_plot, GH, Mu)

        # corrupt training and test data with gaussian noise
        v0 = v0 + (STD) * torch.randn(v0.shape)
        xf = xf + (STD) * torch.randn(xf.shape)

        v0_t = v0_t + (STD) * torch.randn(v0_t.shape)
        xf_t = xf_t + (STD) * torch.randn(xf_t.shape)



    #     '''
    #     LEARNING
    #     '''


        x = v0
        y = xf
        x_t = v0_t
        y_t = xf_t

        def permuteTensors(a, b):
            perm = torch.randperm(a.size(0))
            return (a[perm, :], b[perm, :])

        for (i, m) in enumerate(models):
            torch.save(m.state_dict(), savef[i])
            opts[i] = torch.optim.Adam(m.parameters(), lr=rates[i])

        for t in range(N_epoch):
            trained = False
            # randomly permute data
            (x, y) = permuteTensors(x, y)
            for (i, m) in enumerate(models):
                m.train()
                if not end[i]:
                    trained = True
                    for j in range(N_batches):
                        # get batch
                        samp_j = torch.range(0, N_batch - 1).long() + j * N_batch
                        input_tensor = mdata[i](x[samp_j, :], y[samp_j, :])
                        y_pred = m(mdata[i](x[samp_j, :], y[samp_j, :]))

                        # get loss
                        if i == 0:
                            loss = lossF(y_pred, ldata[i](x[samp_j, :], y[samp_j, :]))
                        else:
                            loss = lossGH(y_pred, ldata[i](x[samp_j, :], y[samp_j, :]), modelGH.gh.item(), modelGH.mu.item())

                        # gradient step
                        opts[i].zero_grad()
                        loss.backward()
                        opts[i].step()
            if not trained:
                break
            if t % 2 == 1:
                print(t, best_loss)
                for (i, m) in enumerate(models):
                    if not end[i]:
                        # update test loss
                        m.eval()
                        y_pred_t = m(mdata_t[i](x_t, y_t))
                        lm = testlosses[i]
                        
                        loss = lm(y_pred_t, ldata_t[i](x_t, y_t))

                        # save the model if it's the best_epoch so far
                        if best_loss[i] - loss.item() > 0.0:
                            best_loss[i] = loss.item()
                            best_epoch[i] = t
                            torch.save(m.state_dict(), savef[i])

                        # terminate tranining if no improvement in (early_stop_epochs) epochs
                        end[i] = end[i] or t - best_epoch[i] > early_stop_epochs[i]

        # save training data
        CVT = torch.cat((v0, xf), dim=1)
        np.savetxt('adam_comp_data.csv', CVT.detach().numpy(), delimiter=',')
    else:
        CVT = torch.tensor(np.loadtxt('adam_comp_data.csv', delimiter=',')).float()
        v0 = CVT[:, 0:1]
        xf = CVT[:, 1:3]
        CV = torch.tensor(np.loadtxt('adam_comp_mods.csv', delimiter=',')).float()
        v0_plot = CV[:, 0:1]
        xf_plot = CV[:, 1:3]

    # reload best models
    for (i, m) in enumerate(models):
        m.load_state_dict(torch.load(savef[i]), strict=False)

    # save models for plotting
    CV = torch.cat((v0_plot, xf_plot), dim=1)
    for m in models:
        m.eval()
        xfp = m(v0_plot)
        CV = torch.cat((CV, xfp), dim=1)
    np.savetxt('adam_comp_mods.csv', CV.detach().numpy(), delimiter=',')


    # '''
    # PLOTTING
    # '''

    # matplotlib settings
    # fm = matplotlib.font_manager.json_load(
        # os.path.expanduser("~/.matplotlib/fontlist-v310.json"))
    # fm.findfont("serif", rebuild_if_missing=True)

    # rc('font', **{'family': ['Computer Modern Roman'], 'size': 10})
    # rc('figure', titlesize=14)
    # rcParams['mathtext.fontset'] = 'cm'
    # rcParams['mathtext.default'] = 'regular'
    # rc('text', usetex=True)
    # rc('legend', fontsize=10)
    # rcParams['text.latex.preamble'] = r'\usepackage{lmodern}'
    # plt.rc('axes', titlesize=14)     # fontsize of the axes title
    # plt.rc('axes', labelsize=14)    # fontsize of the x and y labels



    # styling
    LINEWIDTH = 2
    PENNRED = '#95001a'
    PENNBLUE = '#01256e'
    PENNDGRAY2 = '#44464b'  # noqa
    PENNLGRAY2 = '#e0e1e3'
    PENNLGRAY4 = '#bbbdc0'  # noqa
    PENNYELLOW = '#f2c100'

    colors = [PENNBLUE, PENNRED]

    def styleplot(fig, savefile, width=4, height=4):
        fig.set_size_inches(width, height)
        plt.tight_layout()
        # plt.gcf().subplots_adjust(bottom=0.15)
        fig.savefig(savefile, dpi=400)

    qf = xf[:, 0:2]
    vf = xf[:, 2:4]
    qfD = xf_plot[:, 0:2]
    vfD = xf_plot[:, 2:4]

    # plt.plot(v0_plot[:, 0].detach().numpy(), qfD[:, 0].detach().numpy())
    # for (i, m) in enumerate(models):
    #     m.eval()
    #     xf_pred = m(v0_plot)
    #     qf_pred = xf_pred[:, 0:2]
    #     plt.plot(v0_plot.detach().numpy()[:, 0], qf_pred.detach().numpy()[:, 0], linewidth=LINEWIDTH, color=colors[i], linestyle='dashed')

    # plt.figure()
    # plt.plot(v0_plot[:, 1].detach().numpy(), qfD[:, 1].detach().numpy())
    # for (i, m) in enumerate(models):
    #     m.eval()
    #     xf_pred = m(v0_plot)
    #     qf_pred = xf_pred[:, 0:2]
    #     plt.plot(v0_plot.detach().numpy()[:, 1], qf_pred.detach().numpy()[:, 1], linewidth=LINEWIDTH, color=colors[i], linestyle='dashed')

    # construct legend
    lfinal = ['True System']
    for (i, m) in enumerate(models):
        lfinal = lfinal + legend[i:i + 1]
    lfinal = lfinal + ['Data']

    # plot position predicton
    fig = plt.figure(1)
    fig.suptitle("2D System Predictions")  # Assuming this is a 2D system based on your previous context
    ax1 = plt.subplot(221)
    YMIN = -0.3
    ax1.fill_between(v0_plot[:, 0].detach().numpy(), 0 * v0_plot[:, 0].detach().numpy() + YMIN, color=PENNLGRAY2, label='_nolegend_')
    # ax1.fill_between(v0_plot.numpy(), 0 * v0_plot.numpy() + YMIN, color=PENNLGRAY2, label='_nolegend_')
    plt.plot(v0_plot.detach().numpy()[:, 0], qfD.detach().numpy()[:, 0], linewidth=LINEWIDTH, color=PENNYELLOW)

    for (i, m) in enumerate(models):
        m.eval()
        xf_pred = m(v0_plot)
        qf_pred = xf_pred[:, 0:2]
        plt.plot(v0_plot.detach().numpy()[:, 0], qf_pred.detach().numpy()[:, 0], linewidth=LINEWIDTH, color=colors[i], linestyle='dashed')

    plt.legend(lfinal)

    plt.ylabel(r"Next Position $z'$")
    plt.xlabel(r"Initial Velocity $\dot z$")
    plt.title(r" ")

    ax1 = plt.subplot(223)
    YMIN = -0.3
    plt.plot(v0_plot.detach().numpy()[:, 1], qfD.detach().numpy()[:, 1], linewidth=LINEWIDTH, color=PENNYELLOW)

    for (i, m) in enumerate(models):
        m.eval()
        xf_pred = m(v0_plot)
        qf_pred = xf_pred[:, 0:2]
        plt.plot(v0_plot.detach().numpy()[:, 1], qf_pred.detach().numpy()[:, 1], linewidth=LINEWIDTH, color=colors[i], linestyle='dashed')

    plt.legend(lfinal)

    plt.ylabel(r"Next Position $y'$")
    plt.xlabel(r"Initial Velocity $\dot y$")
    plt.title(r" ")

    # plot velocity prediction
    ax1 = plt.subplot(222)
    plt.plot(v0_plot.detach().numpy()[:, 0], vfD.detach().numpy()[:, 0], linewidth=LINEWIDTH, color=PENNYELLOW)

    for (i, m) in enumerate(models):
        m.eval()
        xf_pred = m(v0_plot)
        vf_pred = xf_pred[:, 2:4]
        plt.plot(v0_plot.detach().numpy()[:, 0], vf_pred.detach().numpy()[:, 0],
                    linewidth=LINEWIDTH, color=colors[i], linestyle='dashed')

    plt.ylabel(r"Next Velocity $\dot z'$")
    plt.xlabel(r"Initial Velocity $\dot z$")
    plt.title(r" ")
    min_v0_plot = torch.min(v0_plot[:, 0]).detach().numpy()
    max_v0_plot = torch.max(v0_plot[:, 0]).detach().numpy()
    plt.gca().set_xlim(min_v0_plot, max_v0_plot)
    ax1 = plt.subplot(224)
    plt.plot(v0_plot.detach().numpy()[:, 1], vfD.detach().numpy()[:, 1], linewidth=LINEWIDTH, color=PENNYELLOW)

    for (i, m) in enumerate(models):
        m.eval()
        xf_pred = m(v0_plot)
        vf_pred = xf_pred[:, 2:4]
        plt.plot(v0_plot.detach().numpy()[:, 1], vf_pred.detach()[:, 1].detach().numpy(),
                    linewidth=LINEWIDTH, color=colors[i], linestyle='dashed')

    plt.ylabel(r"Next Velocity $\dot y'$")
    plt.xlabel(r"Initial Velocity $\dot y$")
    plt.title(r" ")
    min_v0_plot = torch.min(v0_plot[:, 1]).detach().numpy()
    max_v0_plot = torch.max(v0_plot[:, 1]).detach().numpy()
    plt.gca().set_xlim(min_v0_plot, max_v0_plot)
    print("saving figure")
    styleplot(fig, 'PM_config.png', width=10, height=6)

    # # plot loss and loss gradient
    # # construct ground heights
    # NG = 1000
    # GH_SCALE = dt * 10. * STD * SC
    # ghs = torch.linspace(-GH_SCALE, GH_SCALE, NG)
    # l1 = torch.zeros_like(ghs)
    # l2 = torch.zeros_like(ghs)

    # for i in range(NG):

    #     # get L2 loss
    #     mod_gh = GroundHeightQP(ghs[i].clone())
    #     gh = mod_gh.gh
    #     mod_gh.eval()
    #     xf_pred = mod_gh(v0)
    #     l1[i] = lossF(xf, xf_pred).clone().detach()
    #     l2[i] = lossGH(gh, torch.cat((v0, xf), dim=1)).clone().detach()

    # lossdata = torch.cat((ghs.unsqueeze(1), l1.unsqueeze(1), l2.unsqueeze(1)), dim=1)
    # np.savetxt('adam_comp_losses.csv', lossdata.detach().numpy(), delimiter=',')

    # # normalize plots
    # # gl1 /= gl1.max()
    # # gl2 /= gl2.max()
    # # l1 /= l1.max()
    # # l2 /= l2.max()

    # fig = plt.figure(3)
    # plt.plot(ghs.numpy(), l1.detach().numpy(), linewidth=LINEWIDTH, color=PENNBLUE)
    # plt.plot(ghs.numpy(), l2.detach().numpy(), linewidth=LINEWIDTH, color=PENNRED)
    # plt.legend(['L2 Prediction', 'Mechanics-Based (Ours)'])
    # plt.title('1D System Loss')
    # plt.xlabel(r"Modeled Ground Height $\hat z_g$")
    # styleplot(fig, 'PM_loss.png', height=5)
    # plt.show()


@click.command()
@click.option('--regenerate_data/--plot_only', default=True)
def main(regenerate_data: bool):
    generate_figure(regenerate_data)


if __name__ == "__main__": main()
