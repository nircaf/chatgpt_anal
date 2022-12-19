import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from reservoirpy import mat_gen, ESN
from reservoirpy.datasets import mackey_glass
from NeuroBrave_linear_regression import *
from zpalne_plot import *
from multiprocessing import Process, freeze_support

def plot_mackey_glass(X, sample, tau, y, trail):

    t = np.arange(X.shape[0])

    fig = plt.figure(figsize=(13, 5))

    ax1 = plt.subplot((121))
    plt.title(f"Timeserie - {sample} timesteps")
    l1 = ax1.plot(t[:sample], X[:sample], lw=2,
             color="lightgrey", zorder=0)
    l1 = ax1.scatter(t[:sample], X[:sample], c=t[:sample], cmap="viridis", s=6)
    # l2 = ax1.plot(t[:sample], y[:sample], lw=2,
    #               color="greenyellow", zorder=0)
    # ax2 = ax1.twinx()
    # l2 = ax2.plot(t[:sample], trail[:sample], lw=2,
    #          color="deeppink", zorder=0)
    plt.xlabel("$t$")
    plt.ylabel("$P(t)$")

    ax = plt.subplot((122))
    ax.margins(0.05)
    plt.title(f"Phase diagram: $P(t) = f(P(t-\\tau))$")
    plt.plot(X[0: sample], X[tau: sample+tau], lw=1,
             color="lightgrey", zorder=0)
    plt.scatter(X[:sample], X[tau: sample+tau],
             lw=0.5, c=t[:sample], cmap="viridis", s=6)
    plt.xlabel("$P(t-\\tau)$")
    plt.ylabel("$P(t)$")

    cbar = plt.colorbar()
    cbar.ax.set_ylabel('$t$', rotation=270)

    plt.tight_layout()
    plt.suptitle(f'Total duration is: {sample}')
    plt.show(block=False)

def plot_EEG(X, sample, tau):

    t = np.arange(X.shape[0])

    fig = plt.figure(figsize=(13, 5))

    plt.subplot((121))
    plt.title(f"Timeserie - {sample} timesteps")
    plt.plot(t[:sample], X[:sample], lw=2,
             color="lightgrey", zorder=0)
    plt.scatter(t[:sample], X[:sample], c=t[:sample], cmap="viridis", s=6)
    plt.xlabel("$t$")
    plt.ylabel("$P(t)$")

    ax = plt.subplot((122))
    ax.margins(0.05)
    plt.title(f"Phase diagram: $P(t) = f(P(t-\\tau))$")
    plt.plot(X[0: sample], X[tau: sample+tau], lw=1,
             color="lightgrey", zorder=0)
    plt.scatter(X[:sample], X[tau: sample+tau],
             lw=0.5, c=t[:sample], cmap="viridis", s=6)
    plt.xlabel("$P(t-\\tau)$")
    plt.ylabel("$P(t)$")

    cbar = plt.colorbar()
    cbar.ax.set_ylabel('$t$', rotation=270)

    plt.tight_layout()
    plt.show()

def split_timeserie_for_task1(X, forecast, train_length=20000):

    X_train, y_train = X[:train_length], X[forecast: train_length+forecast]
    X_test, y_test = X[train_length: -forecast], X[train_length+forecast:]

    return (X_train, y_train), (X_test, y_test)

def reset_esn(units=100, leak_rate=0.3, spectral_radius=1.25, input_scaling=1.0, density=0.1, input_connectivity=0.2, regularization=1e-8, seed=1221):
    Win = mat_gen.generate_input_weights(units, 1, input_scaling=input_scaling,
                                     proba=input_connectivity, input_bias=True,
                                     seed=seed)

    W = mat_gen.generate_internal_weights(units, sr=spectral_radius,
                                  proba=density, seed=seed)

    reservoir = ESN(leak_rate, W, Win, ridge=regularization)

    return reservoir

def r2_score(y_true, y_pred):
    return 1 - (np.sum((y_true - y_pred[:, 0])**2) / np.sum((y_true - y_true.mean())**2))

def nrmse(y_true, y_pred):
    return np.sqrt((np.sum(y_true - y_pred[:, 0])**2) / len(y_true)) / (y_true.max() - y_true.min())

def train_test_ESN(reservoir=None, X_train=None, y_train=None, X_test=None):
    states = reservoir.train([X_train.reshape(-1, 1)], [y_train.reshape(-1, 1)], return_states=True, verbose=True)

    y_pred, states1 = reservoir.run([X_test.reshape(-1, 1)], init_state=states[0][-1], return_states=True, verbose=True)

    y_pred = y_pred[0].reshape(-1, 1)
    states1 = states1[0]

    return y_pred, states1

def main_ESN(X_many_channels=None, y=None, trail=None):
    freeze_support()
    is_tranj = True
    if X_many_channels is None:
        timesteps = 25000
        tau = 17
        X = mackey_glass(timesteps, tau=tau)

        # rescale between -1 and 1
        X = 2 * (X - X.min()) / (X.max() - X.min()) - 1

        plot_mackey_glass(X, 5000, tau)

        forecast = 10  # for now, predict 10 steps ahead
        (X_train, y_train), (X_test, y_test) = split_timeserie_for_task1(X, forecast)

        sample = 500
        fig = plt.figure(figsize=(15, 5))
        plt.plot(X_train[:sample], label="Training data")
        plt.plot(y_train[:sample], label="True prediction")
        plt.legend()

        plt.show()

        # Prepare the ESN
        units = 100
        leak_rate = 0.3
        spectral_radius = 1.25
        input_scaling = 1.0
        density = 0.1
        input_connectivity = 0.2
        regularization = 1e-8
        seed = 1221

        # reset the net
        reservoir = reset_esn(units=units, leak_rate=leak_rate, spectral_radius=spectral_radius,
                              input_scaling=input_scaling, \
                              density=density, input_connectivity=input_connectivity, \
                              regularization=regularization, seed=seed)

        y_pred, states = train_test_ESN(reservoir=reservoir, X_train=X_train, y_train=y_train, X_test=X_test)

        # show the performance
        sample = 500

        fig = plt.figure(figsize=(15, 7))
        plt.subplot(211)
        plt.plot(np.arange(sample), y_pred[:sample], lw=3, label="ESN prediction")
        plt.plot(np.arange(sample), y_test[:sample], linestyle="--", lw=2, label="True value")
        plt.plot(np.abs(y_test[:sample] - y_pred[:sample]), label="Absolute deviation")
        plt.title(f'$R^2$ score: {r2_score(y_test, y_pred)} and NRMSE: {nrmse(y_test, y_pred)}')

        plt.legend()
        plt.show()
    elif not is_tranj:
        timesteps = 25000
        tau = 17
        fs = 250
        dur_in_sec_one_samp = 1
        for ch_inx, ch in enumerate(X_many_channels.T):
            Mod = ch.shape[0] % (int(fs * dur_in_sec_one_samp))
            X_total = np.reshape(ch[0:-(Mod)], (int(fs * dur_in_sec_one_samp), -1), order='F')
            sample = (int(fs * dur_in_sec_one_samp) - tau - 1)
            agg_signal = []
            R2 = []
            NRMSE = []
            show_iter = 50

            for cur_samp_inx, cur_samp in enumerate(X_total.T):
                print(f'iteration number: #{cur_samp_inx} from: #{X_total.T.shape[0]} iterations at all')
                # X = ch.T
                # rescale between -1 and 1
                cur_samp = 2 * (cur_samp - cur_samp.min()) / (cur_samp.max() - cur_samp.min()) - 1
                agg_signal.append(cur_samp)
                agg_signal_np = np.concatenate(agg_signal)

                cur_train_len = len(agg_signal_np) - (int(0.5 * (fs * dur_in_sec_one_samp) - tau - 1))
                if cur_samp_inx % show_iter == 0:
                    plot_mackey_glass(agg_signal_np, cur_train_len, tau, y, trail[0, :])


                forecast = tau  # for now, predict 10 steps ahead
                (X_train, y_train), (X_test, y_test) = split_timeserie_for_task1(agg_signal_np, forecast, train_length=cur_train_len)

                if cur_samp_inx % show_iter == 0:
                    fig = plt.figure(figsize=(15, 5))
                    plt.plot(X_train[:sample], label="Training data")
                    plt.plot(y_train[:sample], label="True prediction")
                    plt.legend()

                    plt.show()

                # Prepare the ESN
                units = 100
                leak_rate = 0.3
                spectral_radius = 1.25
                input_scaling = 1.0
                density = 0.1
                input_connectivity = 0.2
                regularization = 1e-8
                seed = 1221

                # reset the net
                reservoir = reset_esn(units=units, leak_rate=leak_rate, spectral_radius=spectral_radius,
                                      input_scaling=input_scaling, \
                                      density=density, input_connectivity=input_connectivity, \
                                      regularization=regularization, seed=seed)

                # y_pred, states = train_test_ESN(reservoir=reservoir, X_train=X_train, y_train=y_train, X_test=X_test)
                y_pred, states = train_test_ESN(reservoir=reservoir, X_train=X_train, y_train=y_train,
                                                X_test=X_test)

                R2.append(r2_score(y_test, y_pred))
                NRMSE.append(nrmse(y_test, y_pred))
                # show the performance
                if cur_samp_inx % show_iter == 0:
                    fig = plt.figure(figsize=(15, 7))
                    plt.subplot(211)
                    plt.plot(np.arange(len(y_pred)), y_pred[:, 0], lw=3, label="ESN prediction")
                    plt.plot(np.arange(len(y_test)), y_test, linestyle="--", lw=2, label="True value")
                    plt.plot(np.abs(y_test - y_pred[:, 0]), label="Absolute deviation")
                    plt.title(f'$R^2$ score: {r2_score(y_test, y_pred)} and NRMSE: {nrmse(y_test, y_pred)}')
                    plt.suptitle(f'Total duration is: {len(y_test)}')
                    plt.legend()
                    plt.show()

            fig = plt.figure(figsize=(15, 7))
            plt.subplot(311)
            plt.plot(np.arange(len(R2)), np.asarray(R2), lw=3, label="$R^2$")
            plt.plot(np.arange(len(NRMSE)), np.asarray(NRMSE), lw=3, label="NRMSE")
            plt.legend()
            plt.show()
    elif is_tranj:
        timesteps = 25000
        tau = 17
        fs = 250
        dur_in_sec_one_samp = 1
        for ch_inx, ch in enumerate(X_many_channels.T):
            Mod = ch.shape[0] % (int(fs * dur_in_sec_one_samp))
            X_total = np.reshape(ch[0:-(Mod)], (int(fs * dur_in_sec_one_samp), -1), order='F')
            sample = (int(fs * dur_in_sec_one_samp) - tau - 1)
            agg_signal = []
            agg_linear_proj = []
            agg_phasors = []
            R2 = []
            NRMSE = []
            show_iter = 50

            for cur_samp_inx, cur_samp in enumerate(X_total.T):
                print(f'iteration number: #{cur_samp_inx} from: #{X_total.T.shape[0]} iterations at all in this channel')
                if cur_samp_inx == 0:
                    b0, b1, linear_reg_y, signal_projected, phasors = calc_linear_regression(X=cur_samp, fs=250, signal_processing_duration=1, channels_first=False,
                                           previous_coeff=None, t=None)
                else:
                    b0, b1, linear_reg_y, signal_projected, phasors = calc_linear_regression(X=cur_samp, fs=250,
                                                                                             signal_processing_duration=1,
                                                                                             channels_first=False,
                                                                                             previous_coeff=[b0, b1],
                                                                                             t=None)
                    agg_phasors.append(phasors[0])
                    agg_phasors_np = np.asarray(agg_phasors)
                # X = ch.T
                # rescale between -1 and 1
                cur_samp = signal_projected[0]
                cur_samp = 2 * (cur_samp - cur_samp.min()) / (cur_samp.max() - cur_samp.min()) - 1

                agg_signal.append(cur_samp)
                agg_linear_proj.append(linear_reg_y[0])

                agg_signal_np = np.concatenate(agg_signal)
                agg_linear_proj_np = np.concatenate(agg_linear_proj)



                if cur_samp_inx == 128:
                    break

                cur_train_len = len(agg_signal_np) - (int(0.5 * (fs * dur_in_sec_one_samp) - tau - 1))
                if cur_samp_inx % show_iter == 0:
                    plot_mackey_glass(agg_signal_np, cur_train_len, tau, y, trail[0, :])
                    try:
                        zplane(agg_phasors_np)
                    except:
                        pass
                    fig2 = plt.figure(1222)
                    plt.plot(np.arange(len(agg_linear_proj_np)), agg_linear_proj_np)


                forecast = tau  # for now, predict 10 steps ahead
                (X_train, y_train), (X_test, y_test) = split_timeserie_for_task1(agg_signal_np, forecast, train_length=cur_train_len)

                if cur_samp_inx % show_iter == 0:
                    fig = plt.figure(figsize=(15, 5))
                    plt.plot(X_train[:sample], label="Training data")
                    plt.plot(y_train[:sample], label="True prediction")
                    plt.legend()

                    plt.show(block=False)

                # Prepare the ESN
                units = 512
                leak_rate = 0.3
                spectral_radius = 1.25
                input_scaling = 1.0
                density = 0.1
                input_connectivity = 0.2
                regularization = 1e-8
                seed = 1221

                # reset the net
                reservoir = reset_esn(units=units, leak_rate=leak_rate, spectral_radius=spectral_radius,
                                      input_scaling=input_scaling, \
                                      density=density, input_connectivity=input_connectivity, \
                                      regularization=regularization, seed=seed)

                # y_pred, states = train_test_ESN(reservoir=reservoir, X_train=X_train, y_train=y_train, X_test=X_test)
                y_pred, states = train_test_ESN(reservoir=reservoir, X_train=X_train, y_train=y_train,
                                                X_test=X_test)

                R2.append(r2_score(y_test, y_pred))
                NRMSE.append(nrmse(y_test, y_pred))
                # show the performance
                if cur_samp_inx % show_iter == 0:
                    fig = plt.figure(figsize=(15, 7))
                    plt.subplot(211)
                    plt.plot(np.arange(len(y_pred)), y_pred[:, 0], lw=3, label="ESN prediction")
                    plt.plot(np.arange(len(y_test)), y_test, linestyle="--", lw=2, label="True value")
                    plt.plot(np.abs(y_test - y_pred[:, 0]), label="Absolute deviation")
                    plt.title(f'$R^2$ score: {r2_score(y_test, y_pred)} and NRMSE: {nrmse(y_test, y_pred)}')
                    plt.suptitle(f'Total duration is: {len(y_test)}')
                    plt.legend()
                    plt.show()

            fig = plt.figure(figsize=(15, 7))
            plt.subplot(311)
            plt.plot(np.arange(len(R2)), np.asarray(R2), lw=3, label="$R^2$")
            plt.plot(np.arange(len(NRMSE)), np.asarray(NRMSE), lw=3, label="NRMSE")
            plt.legend()
            plt.show()





if __name__ == '__main__':
    freeze_support()
    main_ESN()
