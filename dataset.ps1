# snake lr-split is added for dataset12
# dataset13 - fixed corrected sa
# dataset14 - add more bridge data
# dataset18 - flip lr g
# dataset19 - reverse2
python dataset.py `
    --drivinglog `
    D:\sdc\driving_logs\t1\center\ `
    D:\sdc\driving_logs\t1\reverse-center\ `
    D:\sdc\driving_logs\t1\reverse2\ `
    D:\sdc\driving_logs\t1\siding\ `
    D:\sdc\driving_logs\t1\siding2\ `
    d:\sdc\driving_logs\t1\snake\ `
    d:\sdc\driving_logs\t1\bridge\ `
    d:\sdc\driving_logs\t1\bridge2\ `
    d:\sdc\driving_logs\t1\g `
    D:\sdc\driving_logs\t2\forward\ `
    D:\sdc\driving_logs\t2\sidings\ `
    --flip-lr `
    D:\sdc\driving_logs\t1\center\ `
    D:\sdc\driving_logs\t1\siding\ `
    D:\sdc\driving_logs\t1\siding2\ `
    d:\sdc\driving_logs\t1\g `
    d:\sdc\driving_logs\t1\snake\ `
    D:\sdc\driving_logs\t2\sidings\ `
    --output dataset19.h5
