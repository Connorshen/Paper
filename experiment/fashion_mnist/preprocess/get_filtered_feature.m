function featureVector = get_filtered_feature(D)

[num_digit, len_digit] = size(D);
featureVector = [];

for k = 1:num_digit
    img = D(k, :)';
    
    Pscale=[2.8 3.6 4.5 5.4 6.3 7.3 8.2 9.2 10.2 11.3 12.3 ];          
    Pwavelength=[3.5 4.6 5.6 6.8 7.9 9.1 10.3 11.5 12.7 14.1 15.4 ];  
    Pfiltersize=[7:2:27];
    %  GENERATE DICTIONARY

    % Layer 1 parameters
    NumbofOrient=16;                                    % Number of spatial orientations for the Gabor filter on the first layer 
    Numberofscales=11;                                   % Number of scales for the Gabor filter on the first laye: must be between 1 and 16.
                                                        % Modify line 7-9 of create_gabors.m to increase to more than than 16 scales
                                                        % Modify line 7-9 of create_gabors.m to increase to more than than 16 scales

    % Layer 2 layer parameters
    maxneigh=floor(8:length(Pscale)/Numberofscales:8+length(Pscale));  % Size of maximum filters (if necessary adjust according Gabor filter sizes)
    L2stepsize=8;                                                      % Step size of L2 max filter (downsampling)

    % LOAD AND DISPLAY GABOR FILTERS
    Gabor=create_gabors(Numberofscales,NumbofOrient,Pscale,Pwavelength,Pfiltersize);
    %displaygabors(Gabor)
    A = reshape(img, 28,28);

    [m n]=size(A);
    L1 = L1_layer(Gabor, A);

    L2 = L2_layer(L1,L2stepsize,maxneigh);

    featureVector(k,:) = L2(:)';
    
end
