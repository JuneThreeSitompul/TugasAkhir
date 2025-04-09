const formatFileSize = (bytes, decimalPoint = 2) => {
    if (bytes == 0) return "0 B";
    
    const si = 1000 // International System of Units (SI)
    const calc = Math.floor(Math.log(bytes) / Math.log(si));
    const size = (bytes / Math.pow(si, calc)).toFixed(decimalPoint);
    const unit = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"][calc];

    return `${parseFloat(size)} ${unit}`;
}

document.addEventListener('DOMContentLoaded', function () {
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    const fileInfo = document.getElementById('file-info');
    const fileDropzone = document.getElementById('file-dropzone');
    const removeFileButton = document.getElementById('file-remove');
    
    fileDropzone.addEventListener('change', function (event) {
        let file = event.target.files[0];
        if (file) {
            fileName.textContent = file.name;
            fileSize.textContent = formatFileSize(file.size);
            fileInfo.classList.remove('hidden');
        }else {
            fileInfo.classList.add('hidden');
        }
    });

    removeFileButton.addEventListener('click', function (event) {
        fileDropzone.value = '';
        fileName.textContent = '';
        fileSize.textContent = '';
        fileInfo.classList.add('hidden');
    });
});