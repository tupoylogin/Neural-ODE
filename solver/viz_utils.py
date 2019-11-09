
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


plt.style.use('seaborn-paper')


def plot_phase_portrait(func, t0=None, xlims=None, ylims=None, num_points=20,
                        xlabel='X', ylabel='Y', ip_rank=None):
    """
    Plots the phase portrait of a system of ODEs containing two dimensions.
    Args:
        func: Must be a callable function with the signature func(t, y)
            where t is a tf.float64 tensor representing the independent
            time dimension and y is a tensor of shape [2] if `ip_rank`
            if not specified, otherwise a tensor of rank = `ip_rank`.
            The function must emit exactly 2 outputs, in any shape as it
            will be flattened.
        t0: Initial timestep value. Can be None, which defaults to a value
            of 0.
        xlims: A list of 2 floating point numbers. Declares the range of
            the `x` space that will be plotted. If None, defaults to the
            values of [-2.0, 2.0].
        ylims: A list of 2 floating point numbers. Declares the range of
            the `y` space that will be plotted. If None, defaults to the
            values of [-2.0, 2.0].
        num_points: Number of points to sample per dimension.
        xlabel: Label of the X axis.
        ylabel: Label of the Y axis.
        ip_rank: Declares the rank of the passed callable. Defaults to rank
            1 if not passed a value. All axis but one must have dimension
            equal to 1. All permutations are allowed, since it will be
            squeezed down to a vector of rank 1.
            Rank 1: Vector output. Shape = [N]
            Rank 2: Matrix output. Shape = [1, N] or [N, 1] etc.
    Returns:
        Nothing is returned. The plot is not shown via plt.show() either,
        therefore it must be explicitly called using `plt.show()`.
        This is done so that the phase plot and vector field plots can be
        visualized simultaneously.
    """

    if xlims is not None and len(xlims) != 2:
        raise ValueError('`xlims` must be a list of 2 floating point numbers')

    if ylims is not None and len(ylims) != 2:
        raise ValueError('`ylims` must be a list of 2 floating point numbers')

    if xlims is None:
        xlims = [-2., 2.]

    if ylims is None:
        ylims = [-2., 2.]

    if ip_rank is None:
        ip_rank = 1

    assert ip_rank >= 1, "`ip_rank` must be greater than or equal to 1."

    x = np.linspace(float(xlims[0]), float(xlims[1]), num=num_points)
    y = np.linspace(float(ylims[0]), float(ylims[1]), num=num_points)

    X, Y = np.meshgrid(x, y)

    u = np.zeros_like(X)
    v = np.zeros_like(Y)

    t = t0 if t0 is not None else 0.0
    t = tf.convert_to_tensor(t, dtype=tf.float64)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            xi = X[i, j]
            yi = Y[i, j]
            inp = tf.stack([xi, yi])

            # prepare input shape for the function
            if ip_rank != 1:
                o = [1] * (ip_rank - 1) + [2]  # shape = [1, ..., 2]
                inp = tf.reshape(inp, o)

            out = func(t, inp)

            # check whether function returns a Tensor or a ndarray
            if hasattr(out, 'numpy'):
                out = out.numpy()

            # reshape the results to be a vector
            out = np.squeeze(out)

            u[i, j] = out[0]
            v[i, j] = out[1]

    Q = plt.quiver(X, Y, u, v, color='black')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plot_vector_field(result, xlabel='X', ylabel='Y'):
    """
    Plots the vector field of the result of an integration call.
    Args:
        result: a tf.Tensor or a numpy ndarray describing the result.
            Can be any rank with excess 1 dimensions. However, the
            final dimension *must* have a rank of 2.
        xlabel: Label of the X axis.
        ylabel: Label of the Y axis.
    Returns:
        Nothing is returned. The plot is not shown via plt.show() either,
        therefore it must be explicitly called using `plt.show()`.
        This is done so that the phase plot and vector field plots can be
        visualized simultaneously.
    """
    if hasattr(result, 'numpy'):
        result = result.numpy()  # convert Tensor back to numpy

    result = np.squeeze(result)

    if result.ndim > 2:
        raise ValueError("Passed tensor or ndarray must be at most a 2D tensor after squeeze.")

    plt.plot(result[:, 0], result[:, 1], 'b-')
    plt.plot([result[0, 0]], [result[0, 1]], 'o', label='start')  # start state
    plt.plot([result[-1, 0]], [result[-1, 1]], 's', label='end')  # end state

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()


def plot_results(time, result, labels=None, dependent_vars=False, **fig_args):
    """
    Plots the result of an integration call.
    Args:
        time: a tf.Tensor or a numpy ndarray describing the time steps
            of integration. Can be any rank with excess 1 dimensions.
            However, the final dimension *must* be a vector of rank 1.
        result: a tf.Tensor or a numpy ndarray describing the result.
            Can be any rank with excess 1 dimensions. However, the
            final dimension *must* have a rank of 2.
        labels: A list of strings for the variable names on the plot.
        dependent_vars: If the resultant dimensions depend on each other,
            then a 2-d or 3-d plot can be made to display their interaction.
    Returns:
        A Matplotlib Axes object for dependent variables, otherwise noting.
        The plot is not shown via plt.show() either, therefore it must be
        explicitly called using `plt.show()`.
    """
    if hasattr(time, 'numpy'):
        time = time.numpy()  # convert Tensor back to numpy

    if hasattr(result, 'numpy'):
        result = result.numpy()  # convert Tensor back to numpy

    # remove excess dimensions
    time = np.squeeze(time)
    result = np.squeeze(result)

    if result.ndim == 1:
        result = np.expand_dims(result, -1)  # treat result as a matrix always

    if result.ndim != 2:
        raise ValueError("`result` must be a matrix of shape [:, 2/3] after "
                         "removal of excess dimensions.")

    num_vars = result.shape[-1]

    # setup labels
    if labels is not None:
        if type(labels) not in (list, tuple):
            labels = [labels]

        if len(labels) != num_vars:
            raise ValueError("If labels are provided, there must be one label "
                             "per variable in the result matrix. Found %d "
                             "labels for %d variables." % (len(labels), num_vars))

    else:
        labels = ["v%d" % (v_id + 1) for v_id in range(num_vars)]

    if not dependent_vars:
        for var_id, var_label in enumerate(labels):
            plt.plot(time, result[:, var_id], label=var_label)

        plt.legend()

    else:
        if num_vars not in (2, 3):
            raise ValueError("For dependent variable plotting, only 2 or 3 variables "
                             "are supported. Provided number of variables = %d" % num_vars)

        if num_vars == 2:
            fig = plt.figure(**fig_args)
            ax = fig.gca()

            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])

            ax.plot(result[:, 0], result[:, 1])

        elif num_vars == 3:
            from mpl_toolkits.mplot3d import Axes3D  # needed for plotting in 3d
            _ = Axes3D

            fig = plt.figure(**fig_args)
            ax = fig.gca(projection='3d')

            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_zlabel(labels[2])

            ax.plot(result[:, 0], result[:, 1], result[:, 2])

        return ax


def get_square_aspect_ratio(plt_axis):
    return np.diff(plt_axis.get_xlim())[0] / np.diff(plt_axis.get_ylim())[0]


def single_feature_plt(features, targets, save_fig=''):
    """Plots a feature map with points colored by their target value. Works for
    2 or 3 dimensions.
    Parameters
    ----------
    features : torch.Tensor
        Tensor of shape (num_points, 2) or (num_points, 3).
    targets : torch.Tensor
        Target points for ODE. Shape (num_points, 1). -1 corresponds to blue
        while +1 corresponds to red.
    save_fig : string
        If string is non empty, save figure to the path specified by save_fig.
    """
    alpha = 0.5
    color = ['red' if targets[i, 0] > 0.0 else 'blue' for i in range(len(targets))]
    num_dims = features.shape[1]

    if num_dims == 2:
        plt.scatter(features[:, 0].numpy(), features[:, 1].numpy(), c=color,
                    alpha=alpha, linewidths=0)
        plt.tick_params(axis='both', which='both', bottom=False, top=False,
                        labelbottom=False, right=False, left=False,
                        labelleft=False)
        ax = plt.gca()
    elif num_dims == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(features[:, 0].numpy(), features[:, 1].numpy(),
                   features[:, 2].numpy(), c=color, alpha=alpha,
                   linewidths=0, s=80)
        ax.tick_params(axis='both', which='both', bottom=False, top=False,
                       labelbottom=False, right=False, left=False,
                       labelleft=False)

    ax.set_aspect(get_square_aspect_ratio(ax))

    if len(save_fig):
        fig.savefig(save_fig, format='png', dpi=200, bbox_inches='tight')
        plt.clf()
        plt.close()
    else:
        plt.show()


def multi_feature_plt(features, targets, save_fig=''):
    """Plots multiple feature maps colored by their target value. Works for 2 or
    3 dimensions.
    Parameters
    ----------
    features : list of torch.Tensor
        Each list item has shape (num_points, 2) or (num_points, 3).
    targets : torch.Tensor
        Target points for ODE. Shape (num_points, 1). -1 corresponds to blue
        while +1 corresponds to red.
    save_fig : string
        If string is non empty, save figure to the path specified by save_fig.
    """
    alpha = 0.5
    color = ['red' if targets[i, 0] > 0.0 else 'blue' for i in range(len(targets))]
    num_dims = features[0].shape[1]

    if num_dims == 2:
        fig, axarr = plt.subplots(1, len(features), figsize=(20, 10))
        for i in range(len(features)):
            axarr[i].scatter(features[i][:, 0].numpy(), features[i][:, 1].numpy(),
                             c=color, alpha=alpha, linewidths=0)
            axarr[i].tick_params(axis='both', which='both', bottom=False,
                                 top=False, labelbottom=False, right=False,
                                 left=False, labelleft=False)
            axarr[i].set_aspect(get_square_aspect_ratio(axarr[i]))
    elif num_dims == 3:
        fig = plt.figure(figsize=(20, 10))
        for i in range(len(features)):
            ax = fig.add_subplot(1, len(features), i + 1, projection='3d')

            ax.scatter(features[i][:, 0].numpy(), features[i][:, 1].numpy(),
                       features[i][:, 2].numpy(), c=color, alpha=alpha,
                       linewidths=0, s=80)
            ax.tick_params(axis='both', which='both', bottom=False, top=False,
                           labelbottom=False, right=False, left=False,
                           labelleft=False)
            ax.set_aspect(get_square_aspect_ratio(ax))

    fig.subplots_adjust(wspace=0.01)

    if len(save_fig):
        fig.savefig(save_fig, format='png', dpi=200, bbox_inches='tight')
        plt.clf()
        plt.close()
    else:
        plt.show()


class Arrow3D(FancyArrowPatch):
    """Class used to draw arrows on 3D plots. Taken from:
    https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def trajectory_plt(model, inputs, targets, timesteps, highlight_inputs=False,
                   include_arrow=False, save_fig=''):
    """Plots trajectory of input points when evolved through model. Works for 2
    and 3 dimensions.
    Parameters
    ----------
    model : anode.models.ODENet instance
    inputs : torch.Tensor
        Shape (num_points, num_dims) where num_dims = 1, 2 or 3 depending on
        augment_dim.
    targets : torch.Tensor
        Shape (num_points, 1).
    timesteps : int
        Number of timesteps to calculate for trajectories.
    highlight_inputs : bool
        If True highlights input points by drawing edge around points.
    include_arrow : bool
        If True adds an arrow to indicate direction of trajectory.
    save_fig : string
        If string is non empty, save figure to the path specified by save_fig.
    """
    alpha = 0.5
    color = ['red' if targets[i, 0] > 0.0 else 'blue' for i in range(len(targets))]
    # Calculate trajectories (timesteps, batch_size, input_dim)
    trajectories = model.odeblock.trajectory(inputs, timesteps)
    # Features are trajectories at the final time
    features = trajectories[-1]

    if model.augment_dim > 0:
        aug = tf.zeros([inputs.shape[0], model.odeblock.odefunc.augment_dim])
        inputs_aug = tf.concat([inputs, aug], 1)
    else:
        inputs_aug = inputs

    input_dim = inputs.shape[-1] + model.augment_dim

    if input_dim == 2:
        # Plot starting and ending points of trajectories
        input_linewidths = 2 if highlight_inputs else 0
        plt.scatter(inputs_aug[:, 0].numpy(), inputs_aug[:, 1].numpy(), c=color,
                    alpha=alpha, linewidths=input_linewidths, edgecolor='orange')
        plt.scatter(features[:, 0].numpy(), features[:, 1].numpy(), c=color,
                    alpha=alpha, linewidths=0)

        # For each point in batch, plot its trajectory
        for i in range(inputs_aug.shape[0]):
            # Plot trajectory
            trajectory = trajectories[:, i, :]
            x_traj = trajectory[:, 0].numpy()
            y_traj = trajectory[:, 1].numpy()
            plt.plot(x_traj, y_traj, c=color[i], alpha=alpha)
            # Optionally add arrow to indicate direction of flow
            if include_arrow:
                arrow_start = x_traj[-2], y_traj[-2]
                arrow_end = x_traj[-1], y_traj[-1]
                plt.arrow(arrow_start[0], arrow_start[1],
                          arrow_end[0] - arrow_start[0],
                          arrow_end[1] - arrow_start[1], shape='full', lw=0,
                          length_includes_head=True, head_width=0.15,
                          color=color[i], alpha=alpha)

        plt.tick_params(axis='both', which='both', bottom=False, top=False,
                        labelbottom=False, right=False, left=False,
                        labelleft=False)

        ax = plt.gca()
    elif input_dim == 3:
        # Create figure
        fig = plt.figure()
        ax = Axes3D(fig)

        # Plot starting and ending points of trajectories
        input_linewidths = 1 if highlight_inputs else 0
        ax.scatter(inputs_aug[:, 0].numpy(), inputs_aug[:, 1].numpy(),
                   inputs_aug[:, 2].numpy(), c=color, alpha=alpha,
                   linewidths=input_linewidths, edgecolor='orange')
        ax.scatter(features[:, 0].numpy(), features[:, 1].numpy(),
                   features[:, 2].numpy(), c=color, alpha=alpha, linewidths=0)

        # For each point in batch, plot its trajectory
        for i in range(inputs_aug.shape[0]):
            # Plot trajectory
            trajectory = trajectories[:, i, :]
            x_traj = trajectory[:, 0].numpy()
            y_traj = trajectory[:, 1].numpy()
            z_traj = trajectory[:, 2].numpy()
            ax.plot(x_traj, y_traj, z_traj, c=color[i], alpha=alpha)

            # Optionally add arrow
            if include_arrow:
                arrow_start = x_traj[-2], y_traj[-2], z_traj[-2]
                arrow_end = x_traj[-1], y_traj[-1], z_traj[-1]

                arrow = Arrow3D([arrow_start[0], arrow_end[0]],
                                [arrow_start[1], arrow_end[1]],
                                [arrow_start[2], arrow_end[2]],
                                mutation_scale=15,
                                lw=0, color=color[i], alpha=alpha)
                ax.add_artist(arrow)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    else:
        raise RuntimeError("Input dimension must be 2 or 3 but was {}".format(input_dim))

    ax.set_aspect(get_square_aspect_ratio(ax))

    if len(save_fig):
        plt.savefig(save_fig, format='png', dpi=400, bbox_inches='tight')
        plt.clf()
        plt.close()
