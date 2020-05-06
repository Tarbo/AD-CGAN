# 
from absl import logging
logging._warn_preinit_stderr = 0
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import os
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import json
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Multiply, BatchNormalization, Activation, Embedding, LeakyReLU
sns.set(style="whitegrid", context="poster")
SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 25
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
pd.options.mode.chained_assignment = None
# plt.xticks(fontsize=30)
# plt.yticks(fontsize=30)
# plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

# set a seed for the generator
np.random.seed(98)


class AD_CGAN():
    def __init__(self, settings_file):
        # load the settings file
        if os.path.isfile(settings_file):
            self.params = json.load(open(settings_file, mode='r'))
        # load parameters from config file
        self.epochs = self.params['epochs']
        self.batch_size = self.params['batch-size']
        self.num_labels = self.params['num-labels']
        self.num_features = self.params['num-features']
        self.input_data_shape = (1, self.num_features)
        self.latent_dim = self.params['latent-dim']
        self.num_units = self.params['hidden-units']

        # define the lists that holds the loss and accuracy history
        self.generator_loss = []
        self.discriminator_loss = []
        self.discriminator_acc = []

        # load the error and optimizer options
        self.mse = tf.keras.losses.MeanSquaredError()
        self.bce = tf.keras.losses.BinaryCrossentropy()
        self.opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)

        # construct the combined model from the generator and discriminator
        self.model_d = self.discriminator()
        self.model_g = self.generator()

        # dont update discriminator weight in the combined gan network
        self.model_d.trainable = False
        # get the generator input to serve as inputs
        latent_input, label_input = self.model_g.input
        # get the generator output
        output_g = self.model_g.output
        # use the label input and generator output as input to discriminator
        output_gan = self.model_d([output_g, label_input])
        self.gan_model = Model(
            inputs=[latent_input, label_input], outputs=output_gan)
        # compile the composite model
        self.gan_model.compile(loss=self.bce, optimizer=self.opt)
        self.gan_model.summary()

        # Load the dataset
        self.X_data, self.Y_data = self.load_data()

    def generator(self):
        # define the parameters of the generator model
        label_input = Input(shape=(1,), dtype='int32', name='label_input')
        latent_input = Input(shape=(self.latent_dim,),
                             name='latent_input')
        label_embedding = Flatten()(Embedding(self.num_labels, self.latent_dim, mask_zero=True,
                                              input_length=1)(label_input))
        conditioned_input = Multiply()([label_embedding, latent_input])
        # first layer
        dense_1 = Dense(self.num_units, input_shape=(
            self.latent_dim,))(conditioned_input)
        activation_1 = LeakyReLU(alpha=0.2)(dense_1)
        # second layer
        dense_2 = Dense(self.num_units*2)(activation_1)
        activation_2 = LeakyReLU(alpha=0.2)(dense_2)
        # third layer
        dense_3 = Dense(self.num_units*4)(activation_2)
        activation_3 = LeakyReLU(alpha=0.2)(dense_3)
        # add batchnormalization
        batch_norm = BatchNormalization(momentum=0.8)(activation_3)
        # get output
        output = Dense(self.num_features, activation='tanh')(batch_norm)
        model = Model(inputs=[latent_input, label_input], outputs=output)
        model.summary()
        return model

    def discriminator(self):
        # define the parameters of the discriminator model
        label_input = Input(shape=(1,), dtype='int32', name='label_input')
        sample_input = Input(shape=(self.num_features,), name='sample_input')
        label_embedding = Flatten()(Embedding(self.num_labels, self.num_features, mask_zero=True,
                                              input_length=1)(label_input))
        # flat_input = Flatten()(sample_input)
        conditioned_input = Multiply()([sample_input, label_embedding])
        # first layer
        dense_1 = Dense(self.num_units*2, input_shape=(
            self.num_features,))(conditioned_input)
        activation_1 = LeakyReLU(alpha=0.2)(dense_1)
        # second layer
        dense_2 = Dense(self.num_units*2)(activation_1)
        activation_2 = LeakyReLU(alpha=0.2)(dense_2)
        dropout_1 = Dropout(0.4)(activation_2)
        # third layer
        dense_3 = Dense(self.num_units*2)(dropout_1)
        activation_3 = LeakyReLU(alpha=0.2)(dense_3)
        dropout_2 = Dropout(0.4)(activation_3)
        # get the output
        target = Dense(1, activation='sigmoid')(dropout_2)
        model = Model(inputs=[sample_input, label_input], outputs=target)
        model.compile(loss=self.bce, optimizer=self.opt, metrics=['accuracy'])
        model.summary()
        return model

    def train(self, sample_interval=50):
        # create a half batch variable
        half_batch = self.batch_size//2
        for epoch in range(self.epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
              # Adversarial ground truths
            y_valid = np.ones((half_batch, 1))
            y_fake = np.zeros((half_batch, 1))
            # Select a random half batch of samples
            idx = np.random.randint(0, self.X_data.shape[0], half_batch)
            samples, labels = self.X_data[idx], self.Y_data[idx]
            # sample half batch of latent samples
            latent_input = np.random.normal(
                0, 1, (half_batch, self.latent_dim))
            # Generate a half batch of new samples
            gen_samples = self.model_g.predict([latent_input, labels])

            # Train the discriminator with half batch of real and fake samples
            d_loss_real = self.model_d.train_on_batch(
                [samples, labels], y_valid)
            d_loss_fake = self.model_d.train_on_batch(
                [gen_samples, labels], y_fake)
            # compte the combined loss for one batch
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # sample a batch of labels for conditioning
            sampled_labels = np.random.randint(
                1, 3, self.batch_size).reshape(-1, 1)
            # sample full batch of latent samples
            latent_input = np.random.normal(
                0, 1, (self.batch_size, self.latent_dim))
            # sample full batch of inverted labels to full the discriminator
            y_valid = np.ones((self.batch_size, 1))
            # Train the generator
            g_loss = self.gan_model.train_on_batch(
                [latent_input, sampled_labels], y_valid)

            # display the progress
            print(
                f'>>> Epoch: {epoch+1}/{self.epochs}\n>>> D loss: {d_loss[0]:.2f}\t D acc: {100*d_loss[1]:.2f} \t G loss: {g_loss} ')
            # save the loss and accuracy of the models for history plot
            self.discriminator_loss.append(d_loss[0])
            self.discriminator_acc.append(100*d_loss[1])
            self.generator_loss.append(g_loss)
            # If at save interval => save generated kdd samples
            if epoch % sample_interval == 0:
                self.evaluate_gan(epoch)
        # save the history to filename
        with open('gen_loss.pkl', 'wb') as f:
            pickle.dump(self.generator_loss, f)
        with open('d_loss.pkl', 'wb') as f:
            pickle.dump(self.discriminator_loss, f)
        with open('d_acc.pkl', 'wb') as f:
            pickle.dump(self.discriminator_acc, f)

    def evaluate_gan(self, epoch, num_samples=400):
        """A function to progressivley check the performance"""
        # load the paths to the images and models directory
        img_path = self.params['img-path']
        models_path = self.params['models-path']

        # Select a random full batch of real samples
        idx = np.random.randint(0, self.X_data.shape[0], self.batch_size)
        real_samples, real_labels = self.X_data[idx], self.Y_data[idx]
        # rescale to [0,1] range
        real_samples = 0.5*real_samples+0.5
        real_labels_cat = np.array(['normal' if var ==
                                    1 else 'malicious' for var in real_labels])
        real_marker = np.array(
            ['r_norm' if var == 'normal' else 'r_mal' for var in real_labels_cat])
        # generate random index
        latent_samples = np.random.normal(0, 1, (num_samples, self.latent_dim))
        fake_labels = np.random.randint(1, 3, size=num_samples)
        # generate the time-series samples
        sample_output = self.model_g.predict([latent_samples, fake_labels])
        # rescale the output to [0,1] range
        fake_sample_output = 0.5*sample_output+0.5
        # change the labels to categorical
        fake_labels_categorical = np.array(['normal' if var ==
                                            1 else 'malicious' for var in fake_labels])
        fake_marker = np.array(
            ['f_norm' if var == 'normal' else 'f_mal' for var in fake_labels_categorical])

        # concat the generated and real samples
        combined_samples = np.concatenate(
            (real_samples, fake_sample_output), axis=0)
        combined_labels = np.concatenate(
            (real_labels_cat, fake_labels_categorical), axis=0)
        combined_marker = np.concatenate((real_marker, fake_marker), axis=0)
        # create the number of perplexities to try for the tSNE
        perplexities = [120, 100, 70, 50, 30, 10]
        # create the subplots
        fig, axes = plt.subplots(len(perplexities), figsize=(10, 36), dpi=300)
        # switch on the axis
        plt.axis('on')
        plt.axis('tight')

        # plot the tsne output of the genrator mixed with real samples
        for idx, perplexity in enumerate(perplexities):
            ax = axes[idx]
            # convert to 2D using tSNE manifold
            tsne = TSNE(n_components=2, random_state=100,
                        perplexity=perplexity)
            tsne_output = tsne.fit_transform(combined_samples)
            # create a dataframe for the generated samples
            data_gen = {
                "x_dim": tsne_output[:, 0], "y_dim": tsne_output[:, 1], "labels": combined_labels, "marker": combined_marker}
            data_gen_df = pd.DataFrame(data=data_gen)

            # use seaborn relplot to visualize the relationship
            ax.set_title(f"Perplexity: {perplexity}")
            ax = sns.scatterplot(x="x_dim", y="y_dim", hue="labels",
                                 style="marker", data=data_gen_df, ax=ax)
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
        # plt.legend()

        # save this file
        abs_path = os.path.dirname(__file__)
        filename = f'generator_quality_plot{epoch}.pdf'
        img_path = os.path.join(abs_path, self.params['img-path'])
        fig.savefig(img_path+filename, dpi=300, papertype='letter',
                    format='pdf', bbox='tight')
        plt.show()
        plt.close()
        # save the generator and discriminator models of this epoch
        models_path = os.path.join(abs_path, self.params['models-path'])
        g_model_name = models_path+f'g_model{epoch}.h5'
        d_model_name = models_path+f'd_model{epoch}.h5'
        self.model_g.save(g_model_name)
        self.model_d.save(d_model_name)
        print(f'>>> Epoch: {epoch}\t Models and Image saved')

    def load_data(self):
        """ process the data into binary class and return the X, Y values"""
        scaler = MinMaxScaler(feature_range=(-1, 1))
        col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
                     "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                     "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                     "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                     "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                     "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                     "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                     "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                     "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                     "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]
        data = pd.read_csv('kddcup.csv', names=col_names)
        # save the label names first
        # labels = pd.unique(data["label"])
        labels_str = data.label.tolist()  # get the string values of label
        # convert to integer labels creaking binary class of normal and anomaly
        labels_int = [1 if key == "normal." else 2 for key in labels_str]
        # change string columns to categorical
        data.label = pd.Series(labels_int)
        data.service = pd.Categorical(data.service)
        data.protocol_type = pd.Categorical(data.protocol_type)
        data.flag = pd.Categorical(data.flag)
        # replace the string columns with codes
        data.service = data.service.cat.codes+1
        data.protocol_type = data.protocol_type.cat.codes+1
        data.flag = data.flag.cat.codes+1
        # convert to float 32
        data = data.astype('float32')
        # select normal samples
        X_normal = data[data["label"] == 1]
        num_samples = X_normal.shape[0]
        # select equal number of anomalous samples to create balance
        X_anomaly = data[data["label"] != 1].sample(
            n=num_samples, random_state=100)
        # concatenate normal and anomalous data and shuffle
        X_data = pd.concat([X_normal, X_anomaly]).sample(frac=1)
        # examine ths shapes
        print(
            f'>>> Original Anomalous shape: {data[data["label"] != 1].shape}')
        print(f'>>> Normal shape:{X_normal.shape}')
        print(f'>>> Anomalous shape: {X_anomaly.shape}')
        print(f'>>> Combined shape: {X_data.shape} ')
        # grab the target column
        target = (X_data.label.values).reshape((X_data.shape[0], 1))
        # drop the label to get X data
        X_data.drop(['label'], axis=1, inplace=True)
        # # scale the data
        X_data = scaler.fit_transform(X_data.values)
        # recreate the dataframe minus the label
        # X_data = pd.DataFrame(X_data, columns=col_names[:-1])
        # check to make sure they have the same shape
        print(f'>>> X shape: {X_data.shape}\tY data: {target.shape} ')
        # save the scaler in case we need it
        # joblib.dump(scaler, self.params["scaler"])
        # X is numpy array and Y is a numpy array
        return X_data, target

    def test_model(self, num_samples, epoch):
        """ A function to test the performance of selected models """
        # load the paths to the images and models directory
        img_path = self.params['test-img']
        models_path = self.params['models-path']

        # Select a random num_samples of the real samples
        assert num_samples < self.X_data.shape[0], "Num of samples index is too large for available real samples"
        idx = np.random.randint(0, self.X_data.shape[0], num_samples)
        real_samples, real_labels = self.X_data[idx], self.Y_data[idx]
        # rescale to [0,1] range
        real_samples = 0.5*real_samples+0.5
        real_labels_cat = np.array(['normal' if var ==
                                    1 else 'malicious' for var in real_labels])
        real_marker = np.array(
            ['real_normal' if var == 'normal' else 'real_malicious' for var in real_labels_cat])
        # generate random index
        latent_samples = np.random.normal(0, 1, (num_samples, self.latent_dim))
        # binary class of either 1 or 2
        fake_labels = np.random.randint(1, 3, size=num_samples)

        # load the model
        abs_path = os.path.dirname(__file__)
        models_path = os.path.join(abs_path, models_path)
        g_model_name = models_path+f'g_model{epoch}.h5'

        # generate the fake samples
        model_g = load_model(g_model_name)
        sample_output = model_g.predict([latent_samples, fake_labels])

        # rescale the output to [0,1] range
        fake_sample_output = 0.5*sample_output+0.5
        # change the labels to categorical
        fake_labels_cat = np.array(['normal' if var ==
                                    1 else 'malicious' for var in fake_labels])
        fake_marker = np.array(
            ['fake_normal' if var == 'normal' else 'fake_malicious' for var in fake_labels_cat])

        # concatenate the generated and real samples
        combined_samples = np.concatenate(
            (real_samples, fake_sample_output), axis=0)
        combined_labels = np.concatenate(
            (real_labels_cat, fake_labels_cat), axis=0)
        combined_marker = np.concatenate((real_marker, fake_marker), axis=0)
        # create the number of perplexities to try for the tSNE
        perplexities = [120, 100]
        # create the subplots
        fig = plt.figure(figsize=(25, 12.5), dpi=600)

        # switch on the axis
        plt.axis('on')
        plt.axis('tight')

        # plot the tsne output of the genrator mixed with real samples
        for idx, perplexity in enumerate(perplexities):
            ax = fig.add_subplot(1, 2, idx+1)
            # convert to 2D using tSNE manifold
            tsne = TSNE(n_components=2, random_state=100,
                        perplexity=perplexity)
            tsne_output = tsne.fit_transform(combined_samples)
            # create a dataframe for the generated samples
            data_gen = {
                "x-axis": tsne_output[:, 0], "y-axis": tsne_output[:, 1], "labels": combined_labels, "marker": combined_marker}
            data_gen_df = pd.DataFrame(data=data_gen)

            # use seaborn scatterplot to visualize the relationship
            ax.set_title(f"Perplexity: {perplexity}")
            ax = sns.scatterplot(x="x-axis", y="y-axis",
                                 hue="labels", style="marker", markers=[".", "P", "*", "X"], data=data_gen_df, ax=ax)
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())

        # save this file
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        filename = f'gen_test_quality{epoch}.pdf'
        img_path = os.path.join(abs_path, img_path)
        fig.savefig(img_path+filename, dpi=600, papertype='letter',
                    format='pdf', bbox='tight')
        plt.show()
        plt.close()

    def classify_points(self, num_samples, epoch):
        """ A function to test the performance of selected models """
        # load the paths to the images and models directory
        # img_path = self.params['test-img']
        models_path = self.params['models-path']

        # generate random index
        latent_samples = np.random.normal(
            0, 1, (num_samples, self.latent_dim))
        # binary class of either 1 or 2
        fake_labels = np.random.randint(1, 3, size=num_samples)

        # load the model
        abs_path = os.path.dirname(__file__)
        models_path = os.path.join(abs_path, models_path)
        g_model_name = models_path+f'g_model{epoch}.h5'

        # generate the fake samples
        model_g = load_model(g_model_name)
        sample_output = model_g.predict([latent_samples, fake_labels])

        # rescale the output to [0,1] range
        fake_sample_output = 0.5*sample_output+0.5

        # Select a random num_samples of the real samples
        idx = np.random.randint(0, self.X_data.shape[0], num_samples)
        real_samples, real_labels = self.X_data[idx], self.Y_data[idx]
        # rescale to [0,1] range
        real_samples = 0.5*real_samples+0.5

        # concatenate the generated and real samples
        combined_samples = np.concatenate(
            (real_samples, fake_sample_output), axis=0)
        # reshape real labels to 1D
        real_labels = np.squeeze(real_labels)
        combined_labels = np.concatenate(
            (real_labels, fake_labels))

        # use 120 perplexity
        perplexity = 120

        # convert to 2D using tSNE manifold
        tsne = TSNE(n_components=2, random_state=100,
                    perplexity=perplexity)
        tsne_output = tsne.fit_transform(combined_samples)
        # create a dataframe for the generated samples
        data_gen = {
            "x": tsne_output[:, 0], "y": tsne_output[:, 1], "labels": combined_labels}
        data_df = pd.DataFrame(data=data_gen)

           # compute mean from the fake output samples only
        # retieve the class data
        fake_data_df = data_df.iloc[num_samples:]
        normal = fake_data_df.loc[fake_data_df['labels'] != 2]
        anomaly = fake_data_df.loc[fake_data_df['labels'] == 2]
        # compute the mean for normal data
        x_ave_n = normal["x"].mean()
        y_ave_n = normal["y"].mean()
        # malicious data mean computation
        x_ave_a = anomaly["x"].mean()
        y_ave_a = anomaly["y"].mean()

        # select the real samples being tested
        data_df.drop(columns="labels", inplace=True)
        test_coords = data_df.iloc[:num_samples]
        # compute the norm to normal cluster
        sub_n_df = test_coords.sub([x_ave_n, y_ave_n], axis='columns')
        norm_n = np.sqrt(np.square(sub_n_df).sum(axis=1))
        # compute the norm to malicious cluster
        sub_a_df = test_coords.sub([x_ave_a, y_ave_a], axis='columns')
        norm_a = np.sqrt(np.square(sub_a_df).sum(axis=1))
        # classify the points
        pred_labels = []
        pred_probs = []
        for n, a in zip(norm_n, norm_a):
            total_dist = np.sum([n, a])
            prob_n = n/total_dist
            prob_a = a/total_dist
            pred_probs.append([prob_n, prob_a])
            if n < a:
                pred_labels.append(1)
            elif a < n:
                pred_labels.append(2)
            elif a == n:
                if np.random.rand() > 0.5:
                    pred_labels.append(2)
                else:
                    pred_labels.append(1)
        return {"target": real_labels, "preds": pred_labels}, pred_probs

# test the train function
# ad_cgan = AD_CGAN('config.json')
# ad_cgan.train()


# 
