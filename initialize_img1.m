function theta = initialize_img1(numK, numM, Train_FM)
   %% Initialize parameters randomly based on layer sizes.
    
    train_x = Train_FM';
    %%  ex1 train a 10 hidden unit SDAE and use it to initialize a FFNN
    %  Setup and train a stacked denoising autoencoder (SDAE)
    rand('state',0)
    sae = saesetup([numM numK]);
    %sae = saesetup([1 1 1 1 1]);
    sae.ae{1}.activation_function       = 'sigm';
    sae.ae{1}.learningRate              = 100;
    sae.ae{1}.inputZeroMaskedFraction   = 0;
    opts.numepochs =   1;
    opts.batchsize = size(train_x,1);
    sae = saetrain(sae, train_x, opts);
    % visualize(sae.ae{1}.W{1}(:,2:end)')

    W1 = sae.ae{1}.W{1}(:,2:end);
    b1 = sae.ae{1}.W{1}(:,1:1);
    W11 = sae.ae{1}.W{2}(:,2:end);
    b11 = sae.ae{1}.W{2}(:,1:1);
    
	% Convert weights and bias gradients to the vector form.
	theta = [W1(:) ;  W11(:) ; b1(:) ; b11(:)];
end