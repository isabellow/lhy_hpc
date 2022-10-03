input = getDirectory("images"); 

birdName = substring(input, 18, lengthOf(input)-1); // change the second input based on the correct file location

run ("Close All");
output = input + "\\" + birdName + " splitData";
File.makeDirectory(output);
File.makeDirectory(output + "\\txtFiles" );

File.makeDirectory(output + "\\tifFiles" );

for (type = 0; type <= 1 ; type++){
    if (type == 0){
       list = getFileList(input + "\\" + birdName + " automatic"); 
    }else {
       list = getFileList(input + "\\" + birdName + " manual");  
    }

for (i=0; i<list.length; i++)                                                          
{
    image = list[i];

    if (type == 0){
       path = input + "\\" + birdName + " automatic\\" + image;
    }else {
       path = input + "\\" + birdName + " manual\\" + image; 
    }

    run("Bio-Formats Macro Extensions");
    Ext.setId(path);
    Ext.getCurrentFile(file);
    run("Bio-Formats Importer", "open=&path autoscale color_mode=Default view=Hyperstack stack_order=XYCZT series_");
    run("Show Info...");

if (i < 9){
leading = "0";
}else{
	leading = ""; 
}

    // save meta data
    if (type == 0){
      saveAs("Text", output + "\\txtFiles\\a" + leading + (i+1) + ".txt"); 
    }else {
      saveAs("Text", output + "\\txtFiles\\m" + leading + (i+1) + ".txt");  
    }
        selectWindow("Info for " + image);
        run("Close");
    getDimensions(width, height, winds, slices, frames);
    
    //  Split channels and save images
    run("Split Channels");     
		for (c = 1; c < = winds; c++)
		{
		selectWindow("C" + c + "-" + image);
            if (type == 0){
                 saveAs("Tiff",  output + "\\tifFiles\\aC" + c + "_" + leading + (i+1));
            }else {
                 saveAs("Tiff",  output + "\\tifFiles\\mC" + c + "_" + leading + (i+1)); 
           }
		}
       run ("Close All");
 }
}