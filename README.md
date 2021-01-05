# ContactNets
This repository contains source code for the paper [ContactNets: Learning Discontinuous Contact Dynamics with Smooth, Implicit Representations](https://arxiv.org/abs/2009.11193) by [Samuel Pfrommer\*](http://sam.pfrommer.us/), [Mathew Halm\*](https://www.grasp.upenn.edu/people/mathew-halm), and [Michael Posa](https://www.grasp.upenn.edu/people/michael-posa), published in [CoRL 2020](https://www.robot-learning.org/program/accepted-papers).

## Attribution notes
The osqpth and lemkelcp libraries found in lib are not our own and are protected by the Apache Software License and MIT licenses, respectively. The file `contactnets/utils/quaternion.py` contains code extended from the Facebook research `QuaterNet` project, protected under Creative Commons Attribution-NonCommercial license. References to these projects are found below.

* https://github.com/oxfordcontrol/osqpth
* https://github.com/AndyLamperski/lemkelcp
* https://github.com/facebookresearch/QuaterNet 

# Setup
## Requirements
* Python 3.6.9 or 3.7. Running `python --version` and `pip --version` in the command line both should satisfy this requirement and correspond to the same python install.
* 16GB RAM
* Linux (tested on Ubuntu 18.04 LTS)

No GPU is required. These instructions were tested on a fresh no-GPU Google Cloud [Deep Learning VM](https://console.cloud.google.com/marketplace/details/click-to-deploy-images/deeplearning) instance.

## Dependencies
Due to the large number of dependencies, installation instructions are written for install in a virtual environment:
```
python -m venv cnets_env
source ./cnets_env/bin/activate
```

The venv must be re-sourced after reboots. Once in the environment, all python prereqs and local code can be installed by running
```
chmod u+x ./setup.sh
./setup.sh
```

Additionally, the following linux packages must be installed:
```
sudo apt-get install freeglut3-dev psmisc
```

If you're getting errors relating to "no such file or directory 'tensorboard'" the tensorboard command might not be in your path. Check by just running *tensorboard* in the terminal. If the command can't be found, you might have to add a symbolic link to */usr/bin*:

```
sudo ln -s ~/.local/bin/tensorboard /usr/bin
```

If you get *grpcio* errors you might also need to run:


```
pip install --upgrade grpcio
```

# Generating Figure 1

The code used to generate Figure 1 from the paper can run by
```
python figure1.py
```
The figure will be generated as `PM_config.png` and `PM_loss.png`.

# Executing ContactNets
After installing the above dependencies, execute
```
python experiment.py --method e2e --tosses 100
```
`method` can be chosen to be one of either `e2e`, `polytope`, or `deep`, and tosses represents the number of training tosses (x-axis in Figure 5 of the accompanying submission). In order to verify that the install is working correctly it might be helpful to first run the `e2e` method with `tosses=1`.

You can view the training process on tensorboard at `localhost:6006`. The `images` tab contains a log-scaled plot of various losses / regularizers, rendering rollouts of a subset of the tosses, and for `ContactNets` methods renderings of the learned `phi` functions over configuration space; here `theta` represents the angle of rotation around the y-axis. All `g_normal_x/y/z` plots render a projection of `ContactNets` vertex positions for `phi_n` onto the corresponding axis. Units are scaled such that with enough training samples, the corner positions should eventually converge to `(1,1,1),(1,1,-1),...`, even with as few as 30 training tosses. `h_tangent_x/y/z` represents the same idea for `phi_t`, and generally needs 60-100 training tosses to converge nicely, although due to friction irregularities will not converge as precisely as the normal component.

Different losses are plotted in the `Custom Scalars` tab. Trajectory position integral error corresponds to plot 5a, trajectory angle integral error corresponds to plot 5b, and trajectory penetration integral corresponds to plot 5c. For `ContactNets` methods `Surrogate` losses refer to the loss in equations 16-17, while for `e2e` the `vel_basic` losses refer to the equation 21. After `patience` epochs have passed without an improvement on the validation loss, summary statistics are outputted to `out/best/stats.json`.

## Headless server
If you are running this on a headless linux server, you will need to install the xvfb package and then execute the following commands before running `experiment.py`:
`
Xvfb :5 -screen 0 800x600x24 &
export DISPLAY=:5
`

You may see some periodic ALSA lib errors; these are generally harmless.

## Data notes
Processed data is included in `out/data/all`. Since raw data files are too large for GitHub, please contact Samuel Pfrommer or Mathew Halm at the emails listed in the CoRL publication if you are interested in obtaining access.

# Architecture
## Simulation
An **Entity** represents something in your environment, whether it is a polytope object, ground, or point mass. They keep track of a history of configurations, velocities, and control impulses for that body. Something without a real state, like a ground entity, maintains state vectors of length zero. Anything inheriting from entity must specify the dimensions of its configuration / velocity, as well as implement methods specifying its mass matrix, gamma, and free evolution dynamics.


A set of entities are related by an **Interaction**. Right now, interactions only act on pairs of entities. Something extending **Interaction** must implement methods for computing *phi*, *Jn*, *phi_t*, *Jt_tilde*, and *k*, where *k* is the number of elements in *phi* (for current approaches, interpretable as the number of vertices). There are a few interaction implementations currently, including polygon-ground 2D/3D interactions. These are hard-coded and are mainly used for simulation. For learning, each experiment generally has some kind of "learnable" interaction, which subclasses one of the basic ones but adds trainable parameters. For example, you could have a learnable poly-ground interaction which subclasses the basic poly-ground interaction, retains its tangent jacobian calculations, but uses a deep network to compute *phi* and *Jn*.


All interactions are managed by a single **InteractionResolver**. This object resolves all interactions in the environment for each step and computes the next states of all entities registered to it. Currently implemented resolvers are a LCP Stewart-Trinkle based resolver, an elastic Anitescu-Potra resolver, and a resolver for **DirectInteractions**, which don't learn any special parameterization but instead directly learn forces between objects (these are what we refer to as end-to-end methods). An important thing to note is that there should not be any trainable parameters introduced at the resolver level. **Interaction** and **Entity** instances should be the only things containing learnable parameters.


Finally, we have a **System** which ties the above components together. A **System** consists of a single resolver and a list of entities. Systems subclass PyTorch modules, so this system inherets the parameters of all the entities as well as the interactions (which first get inhereted by the resolver). The **System** simulates a rollout of a model by taking as input a list of list of controls, with each entity getting one control per time step. The system will then use the resolver to compute inter-body impulses and allow each entity to compute its own dynamics step, combining the control variable (which was assigned to its history) and the resolved collision impulses.

## Training
As hinted at above, the **System** is now a PyTorch module with all the trainable parameters of the system, whether those be in interactions or entities. Parameters can be added to either, depending on the needs of the researcher; parameters relating to entities will simply "shared" between its interactions (e.g., mass or inertia).


A **Loss** operates on a system to produce a scalar loss, working from the current configuration / velocity / control histories stored in the entities. Losses can either operate stepwise or trajectorywise; if they are stepwise, they expect to compute a loss over a system with a history of length two. Additionally, some losses are potentially allowed to mutate the system, which is only allowed for reporting losses (we'll see what that is later).


A **LossManager** manages a list of training losses and reporting losses. Training losses are what are actually used to compute the gradient, and must be non-mutating (since all the losses are evaluated over a single system state). Reporting losses are things like trajectory error / l2 cost which are outputted as metrics but do not need to provide gradients; these can mutate the system if necessary.


The **Trainer** class manages the training process. It has as member variables a **LossManager**, **DataManager**, and **TensorboardManager**, which more or less do what you would expect. It also has a few callbacks that allow modification of the gradients before and after each traning step if required by certain methods.
