

function show_error_image(init_para, testing_result)
    
    [ind_digit_data, digit_data] = get_testdata(init_para.digit_label);
    
    i_label = testing_result(1, 1);
    reward = testing_result(1, 4);
    if reward == 0
        
        data_img = digit_data(i_label, :);
        X = reshape(data_img,28,28);
        [r,c] = size(X);                           %# Get the matrix size
        imagesc((1:c)+0.5,(1:r)+0.5,X');            %# Plot  image
        colormap(gray);                              %# Use a gray colormap
        axis equal 
    end
    