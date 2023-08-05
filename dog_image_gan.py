import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose, Reshape, Flatten
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
import os
import matplotlib.image as mpimg

def define_gan(g_model, d_model):
    # Make weights in the discriminator not trainable
    d_model.trainable = False
    # Connect the generator and discriminator
    gan_output = d_model(g_model.output)
    model = Model(g_model.input, gan_output)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    print('\\nGAN Model:')
    model.summary()
    return model
# Define the standalone discriminator model
def define_discriminator(in_shape=(32,32,3)):
    model = Sequential()
    model.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('Discriminator Model:')
    model.summary()
    return model

# Define the standalone generator model
def define_generator(latent_dim):
    model = Sequential()
    model.add(Dense(256 * 4 * 4, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4, 4, 256)))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
    print('\\nGenerator Model:')
    model.summary()
    return model


# Load and prepare cifar10 training images
def load_real_samples():
    # load cifar10 dataset
    (trainX, trainY), (_, _) = cifar10.load_data()
    # select all of the 'cat' class
    selected_ix = trainY.flatten() == 3
    X = trainX[selected_ix]
    # convert from unsigned ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    return X


# Select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = np.random.randint(0, dataset.shape[0], n_samples)
	# select images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = np.ones((n_samples, 1))
	return X, y

# Generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = np.random.randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# Use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = g_model.predict(x_input)
	# create 'fake' class labels (0)
	y = np.zeros((n_samples, 1))
	return X, y

# Create and save a plot of generated images
def save_plot(examples, epoch, n=7):
	# scale from [-1,1] to [0,1]
	examples = (examples + 1) / 2.0
	# plot images
	for i in range(n * n):
		plt.subplot(n, n, 1 + i)
		plt.axis('off')
		plt.imshow(examples[i])
	# save plot to file
	filename = 'generated_plot_e%03d.png' % (epoch+1)
	plt.savefig(filename)
	plt.close()

# Evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
	# prepare real samples
	X_real, y_real = generate_real_samples(dataset, n_samples)
	# evaluate discriminator on real examples
	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
	# prepare fake examples
	x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
	# evaluate discriminator on fake examples
	_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	# save plot
	save_plot(x_fake, epoch)
	# save the generator model tile file
	filename = 'generator_model_%03d.h5' % (epoch+1)
	g_model.save(filename)
        
# Function to evaluate the model
def evaluate_model(history, g_model, d_model, dataset, latent_dim, n_samples=100):
    print("Evaluating Model...")

    # 1. Discriminator Accuracy
    X_real, y_real = generate_real_samples(dataset, n_samples)
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    print(f"Discriminator Accuracy Real: {acc_real * 100:.2f}% Fake: {acc_fake * 100:.2f}%")

    # 2. Generated Sample Visualization
    print("Generated Images:")
    examples, _ = generate_fake_samples(g_model, latent_dim, n_samples=16)
    for i in range(16):
        plt.subplot(4, 4, 1 + i)
        plt.axis('off')
        plt.imshow((examples[i] + 1) / 2.0)
    plt.show()

    # 3. Loss Plot
    plt.plot(history['d_loss'], label='Discriminator Loss')
    plt.plot(history['g_loss'], label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=200, n_batch=128):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    
    # Lists to track average losses for each epoch
    avg_d_losses = []
    avg_g_losses = []
    
    # manually enumerate epochs
    for i in range(n_epochs):
        d_losses = []
        g_losses = []
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # create training set for the discriminator
            X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
            # update discriminator model weights
            d_loss, _ = d_model.train_on_batch(X, y)
            d_losses.append(d_loss)
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = np.ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            g_losses.append(g_loss)
            # summarize loss on this batch
            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
        
        # Average loss for the epoch
        avg_d_losses.append(np.mean(d_losses))
        avg_g_losses.append(np.mean(g_losses))
        print(f'Epoch {i+1}: Avg D Loss: {avg_d_losses[-1]:.3f}, Avg G Loss: {avg_g_losses[-1]:.3f}')

    history = {
        'd_loss': avg_d_losses,
        'g_loss': avg_g_losses
    }
    return history

    

def display_generated_images():
    # Path pattern for the generated images
    path_pattern = 'generated_plot_e%03d.png'

    # Number of epochs to display (e.g., every 10th epoch)
    display_epochs = [10, 50, 100, 150, 200]

    plt.figure(figsize=(15, 5))
    for i, epoch in enumerate(display_epochs):
        file_path = path_pattern % epoch
        if os.path.exists(file_path):
            img = mpimg.imread(file_path)
            plt.subplot(1, len(display_epochs), i + 1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'Epoch {epoch}')
        else:
            print(f"File {file_path} not found.")
    plt.show()
    
	
def main():
    # Size of the latent space
    latent_dim = 100
    
    # Create the discriminator
    d_model = define_discriminator()
    
    # Create the generator
    g_model = define_generator(latent_dim)

    # Create the GAN
    gan_model = define_gan(g_model, d_model)
    
    # Load CIFAR-10 data (only cat images)
    dataset = load_real_samples()
    
   # Train the model
    history = train(g_model, d_model, gan_model, dataset, latent_dim)

    # Display generated images
    display_generated_images()

    # Evaluate the model
    evaluate_model(history, g_model, d_model, dataset, latent_dim)

if __name__ == "__main__":
    main()