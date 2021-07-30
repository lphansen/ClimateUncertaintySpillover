function [derivative_array] = derivative_computer(array)
    array_reform = [0; array(1:end-1,:)];
    derivative_array_temp = array - array_reform;
    derivative_array = derivative_array_temp(2:end,:);
end

