function ret = soroll(img, n)

if n == 1
    ret = imnoise(img, 'gaussian');

elseif n == 2
    ret = imnoise(img, 'salt & pepper');
   
else
    ret = imnoise(img, 'poisson');

end