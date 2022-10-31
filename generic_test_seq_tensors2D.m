function [X2, Y2s, Y2, Yh2, Yhs2, Bti, Bto, Sx2, Sy2, k_tob] = generic_test_seq_tensors2D(M, x_in, t_in, y_out, t_out, l_sess, l_test, t_sess, sess_off, offset,...
    norm_fli, norm_flo, Bi, Bo, k_ob, k_tob)

    %% Test regression ANN
    if(k_tob == 0)
        [m,~] = size(M);
        k_tob = floor(l_sess/t_out); %ceil
        if(l_sess + k_tob*t_out*(t_sess-sess_off)) > m
            k_tob = k_tob - 1;
        end
    end

    n_xy = x_in + y_out; 

    X2 = ones([n_xy, k_ob+1, k_tob, t_sess-sess_off]);
    Y2s = zeros([n_xy, k_ob+1, k_tob, t_sess-sess_off]);
    Y2 = zeros([n_xy, t_out, k_tob, t_sess-sess_off]);
    Yh2 = zeros([n_xy, t_out, k_tob, t_sess-sess_off]);
    Yhs2 = zeros([n_xy, t_out, k_tob, t_sess-sess_off]);
    Bti = zeros([4, n_xy, k_tob, t_sess-sess_off]);
    Bto = zeros([4, n_xy, k_tob, t_sess-sess_off]);

    %Segment boundaries
    Sx2 = zeros([2, k_tob, t_sess-sess_off]);
    Sy2 = zeros([2, k_tob, t_sess-sess_off]);


    % Re-format test input into session tensor
    for i = 1:t_sess-sess_off
        for j = 1:k_tob
            % extract and scale observation sequence
            idx = (i+sess_off)*l_sess + (j-1)*t_out + 1 + offset - k_ob;

            st_idx = idx-1;
            end_idx = idx+k_ob-1;
            Sx2(1,j,i) = st_idx;
            Sx2(2,j,i) = end_idx;

            Mxw = M(idx-1:idx+k_ob-1, 1:n_xy);
            % scale bounds over observation span
            [Bti(1,:,j,i), Bti(2,:,j,i)] = bounds(Mxw,1);
            Bti(3,:,j,i) = mean(Mxw,1);
            Bti(4,:,j,i) = std(Mxw,1);

            %Mx = reshape( Mxw', [m_in,1] );
            X2(:, :, j, i) = Mxw';


            st_idx = idx+k_ob;
            end_idx = idx+k_ob+t_out-1;
            Sy2(1,j,i) = st_idx;
            Sy2(2,j,i) = end_idx;

            Myw = M(idx+k_ob:idx+k_ob-1+t_out, 1:n_xy);
            % scale bounds over observation span
            [Bto(1,:,j,i), Bto(2,:,j,i)] = bounds(Myw,1);
            Bto(3,:,j,i) = mean(Myw,1);
            Bto(4,:,j,i) = std(Myw,1);

            %Mx = reshape( Mxw', [m_in,1] );
            Yh2(:, :, j, i) = Myw';
            Yhs2(:, :, j, i) = Myw';
        end


        if(norm_fli)
            for j = 1:k_tob
                % extract and scale observation sequence
                idx = (i+sess_off)*l_sess + (j-1)*t_out + 1 + offset - k_ob;


                Mxw = M(idx-1:idx+k_ob-1, 1:n_xy);
                % bounds over session
                %MinSessi = Bi(1,:,i); 
                %MaxSessi = Bi(2,:,i);
                MeanSessi = Bi(3,:,i);
                StdSessi = Bi(4,:,i);

                Mxw = generic_mean_std_scale2D(Mxw, MeanSessi, StdSessi);
                %Mxw = generic_minmax_scale2D(Mxw, MinSessi, MaxSessi);
                
                X2(:, :, j, i) = Mxw';
            end
        end


        if(norm_flo)
            for j = 1:k_tob
                % extract and scale observation sequence
                idx = (i+sess_off)*l_sess + (j-1)*t_out + 1 + offset - k_ob;


                Myw = M(idx+k_ob:idx+k_ob-1+t_out, 1:n_xy);
                % bounds over session
                %MinSesso = Bo(1,:,i); 
                %MaxSesso = Bo(2,:,i);
                MeanSesso = Bo(3,:,i);
                StdSesso = Bo(4,:,i);

                Myw = generic_mean_std_scale2D(Myw, MeanSesso, StdSesso);
                %Myw = generic_minmax_scale2D(Myw, MinSesso, MaxSesso);
                
                Yhs2(:, :, j,i) = Myw';
            end
        end

    end

end