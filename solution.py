import os
import typing
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.kernel_approximation import Nystroem

# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = True
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation
EVALUATION_GRID_POINTS_3D = 50  # Number of points displayed in 3D during evaluation


# Cost function constants
COST_W_UNDERPREDICT = 25.0
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 10.0



#todolist

# custom prediction that combines cost and probability distribution
# check marginal likelyhood (done default by sklearn gpr)
# check all the kernels and the hyper parameters
#rbf matern exponential dotproduct whitenoise constant etc 
# low rank approx
# ensamble method (base on max marginal likelyhood by region?)
# local gp approx
# undersampling rnd or using cluster
# kernel low rank approx nystrom method random fourier features.

# baseline 0:  simple rbf kernel , with random under sampling if takes too long 5185.452 2081.279
 
# baseline 1:  different kernels hyperparameter testing framework , same low rank approx. 
# baseline 2:  rbf kernel, nystrom or fourier transform for low rank approx    
# baseline 3:  all the above with custom prediction.
# ensamble / local gp approx.

#divided by zero guassian kernel runtime error
# try fix value
class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)
        # TODO: Add custom initialization for your model here if necessary
        self.gpr = None

    def make_predictions(self, test_features: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of locations.
        :param test_features: Locations as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        # TODO: Use your GP to estimate the posterior mean and stddev for each location here
        gp_mean,gp_std = self.gpr.predict(test_features,return_std=True)
        

        # TODO: Use the GP posterior to form your predictions here
        predictions = gp_mean+0.2*gp_std
        # can do a arppoximation.
        #predictions = gp_mean
        return predictions, gp_mean, gp_std

    def fitting_model(self, train_GT: np.ndarray,train_features: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_features: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_GT: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """
        # TODO: Fit your model here
        baseline =1.0
        if(baseline < 1):
            num_samples = train_GT.shape[0]
            if(baseline ==0.0):
                kernel = DotProduct() + WhiteKernel() #37422.642
            elif(baseline ==0.1):
                kernel =  ConstantKernel(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) * RBF(length_scale=0.5, length_scale_bounds=(0.0, 100.0))
            print(train_GT.shape)
            mask = np.random.choice(np.arange(train_GT.shape[0]),int(num_samples*0.1))
            self.gpr = GaussianProcessRegressor(kernel=kernel).fit(train_features[mask,:],train_GT[mask])
        if baseline ==2:
            kernel =  ConstantKernel(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) * RBF(length_scale=0.5)+ WhiteKernel()
            feature_map_nystroem = Nystroem(kernel='rbf',gamma=.2,
                                 random_state=1,
                                 n_components=500)
            data_trans = feature_map_nystroem.fit_transform(train_features)
            print(data_trans.shape)
            #self.gpr = GaussianProcessRegressor(kernel=kernel).fit(data_trans,train_GT)
            
        if baseline <2 and baseline >=1:
            num_samples = train_GT.shape[0]
            first = True
            #todo: try multiplying linear kernel with rbf and matern
            #kernel =  ConstantKernel(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) * RBF(length_scale=0.5)+RBF(length_scale=2)+WhiteKernel()
            #print(train_GT.shape)
            values=[5]
            mask = np.random.choice(np.arange(train_GT.shape[0]), 1000)
            for i in values:
                # kernel =  ConstantKernel(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) * RBF(length_scale=i)+RBF(length_scale=2)+WhiteKernel()
                kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) * Matern(length_scale=i,length_scale_bounds=(0.001,1000),nu=1.5)+WhiteKernel()
                gpr_candidate = GaussianProcessRegressor(kernel=kernel).fit(train_features[mask, :], train_GT[mask])

                cur_log_p= gpr_candidate.log_marginal_likelihood
                print(i,cur_log_p)
                if(first):
                    best_setting = i
                    best_logp = cur_log_p
                    best_model = gpr_candidate
                elif cur_log_p>best_logp:
                    best_logp = cur_log_p
                    best_model =  gpr_candidate
                    best_setting = i
            self.gpr = best_model
            print("best_setting ",best_setting)

                #kernel = DotProduct()* RBF(length_scale=i)+WhiteKernel()


def cost_function(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask_1 = predictions < ground_truth
    weights[mask_1] = COST_W_UNDERPREDICT

    # Case ii): significant overprediction
    mask_2 = (predictions >= 1.2*ground_truth)
    weights[mask_2] = COST_W_OVERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)



def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')
    fig = plt.figure(figsize=(30, 10))
    fig.suptitle('Extended visualization of task 1')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.make_predictions(visualization_xs)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_stddev = np.reshape(gp_stddev, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0
    vmax_stddev = 35.5

    # Plot the actual predictions
    ax_predictions = fig.add_subplot(1, 3, 1)
    predictions_plot = ax_predictions.imshow(predictions, vmin=vmin, vmax=vmax)
    ax_predictions.set_title('Predictions')
    fig.colorbar(predictions_plot)

    # Plot the raw GP predictions with their stddeviations
    ax_gp = fig.add_subplot(1, 3, 2, projection='3d')
    ax_gp.plot_surface(
        X=grid_lon,
        Y=grid_lat,
        Z=gp_mean,
        facecolors=cm.get_cmap()(gp_stddev / vmax_stddev),
        rcount=EVALUATION_GRID_POINTS_3D,
        ccount=EVALUATION_GRID_POINTS_3D,
        linewidth=0,
        antialiased=False
    )
    ax_gp.set_zlim(vmin, vmax)
    ax_gp.set_title('GP means, colors are GP stddev')

    # Plot the standard deviations
    ax_stddev = fig.add_subplot(1, 3, 3)
    stddev_plot = ax_stddev.imshow(gp_stddev, vmin=vmin, vmax=vmax_stddev)
    ax_stddev.set_title('GP estimated stddev')
    fig.colorbar(stddev_plot)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def main():
    # Load the training dateset and test features
    train_features = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_GT = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_features = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Fit the model
    print('Fitting model')
    model = Model()
    model.fitting_model(train_GT,train_features)

    # Predict on the test features
    print('Predicting on test features')
    predictions = model.make_predictions(test_features)
    print(predictions)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
