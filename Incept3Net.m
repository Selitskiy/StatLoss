classdef Incept3Net < Incept3Layers & BaseClassNet & IDSInputClassNet

    properties

    end

    methods
        function net = Incept3Net(n_out, ini_rate, max_epoch, mb_size)

            net = net@Incept3Layers();
            net = net@BaseClassNet(n_out, ini_rate, max_epoch);
            net = net@IDSInputClassNet(mb_size);

            net.name = "InceptionV3";

        end


        %function [net, X, Y, Bi, Bo, Sx, Sy, k_ob] = TrainTensors(net, M, l_sess, n_sess, norm_fli, norm_flo)

        %    [net, X, Y, Bi, Bo, Sx, Sy, k_ob] = TrainTensors@MLPInputNet2D(net, M, l_sess, n_sess, norm_fli, norm_flo);

        %    net = Create(net);

        %end



        %function net = Train(net, inputDS)
        %    fprintf('Training %s net\n', net.name); 
        %
        %    net = Train@IDSInputClassNet(net, inputDS);
        %end

        
    end
end