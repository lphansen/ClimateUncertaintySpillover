clear
candlebar_data = table2array(readtable("all.csv"));
terminal = 10;
x = 1:terminal;
tol =0.5;
open_close_diff = candlebar_data(:,2) - candlebar_data(:,5);

candlebar_derivative=diff(open_close_diff);

candlebar_indicator = double(candlebar_derivative >0);
plot(x, candlebar_indicator(1:terminalgii))





