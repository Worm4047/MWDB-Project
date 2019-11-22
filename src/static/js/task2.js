$(document).ready(function(){
    console.log("Let's go !");
    // console.log($('#dorsal_clusters'))
});

function show_dorsal_clusters(){
    $('#palmar_clusters').hide();
    $('#query_clusters').hide();
    $('#dorsal_clusters').show();
};
function show_palmar_clusters(){
    $('#dorsal_clusters').hide();
    $('#query_clusters').hide();
    $('#palmar_clusters').show();
};
function show_query_results(){
    $('#dorsal_clusters').hide();
    $('#palmar_clusters').hide();
    $('#query_clusters').show();
    
}