classdef BaseClassNet

    properties
        name = [];

        %m_in
        n_out

        ini_rate 
        max_epoch
    
        lGraph = [];
        options = [];
        trainedNet = [];
    end

    methods
        function net = BaseClassNet(n_out, ini_rate, max_epoch)

            net.n_out = n_out;

            net.ini_rate = ini_rate;
            net.max_epoch = max_epoch;

        end

    end
end