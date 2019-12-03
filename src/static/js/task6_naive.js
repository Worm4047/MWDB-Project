$(document).ready(function(){
    console.log("Let's go");
    $('.images').on('click', 'div', function(event) {

        console.log("Clicked");
        console.log(event);
        $target = $(event.target);
        $target.removeClass('irrelevant');
           $target.toggleClass('relevant');
    
    });

    $('.images').on('contextmenu', 'div', function(event) {
        console.log(event);
        event.preventDefault();
        $target = $(event.target);
        $target.removeClass('relevant');

        $target.toggleClass('irrelevant');
    });
});





function change_images(images){
    console.log("Adding images");
    var $imageRows = $('.images');
    // $imageRows.empty();
    images.forEach(img => {
        console.log(img);
        $cardDiv = $('div');
        $cardDiv.addClass('card');
        $img = $('img');
        $img.attr("src", img);
        $img.attr("alt", img);
        $img.attr("data", img);
        $img.addClass('card-img-top')
        console.log('Appending img to card');
        $cardDiv.append($img);
        console.log('Appending card to row');
        $imageRows.add($cardDiv);
        
    }); 

    console.log(images);
}

iterationCount = 0
$('#submitButton').on('click', function(event){
    iterationCount++;
    var relevant = []
    var nonrelevant = [];
    var all = $("img").map(function() {
        if($(this).hasClass("relevant"))
            relevant.push($(this).attr('src'));
        else if($(this).hasClass("irrelevant"))
            nonrelevant.push($(this).attr('src'));

        console.log(this.className, $(this).attr('src'));
    }).get();
    console.log(data);

    var data = {'relevant' : relevant, 'nonrelevant' : nonrelevant};
    jQuery.ajax({
        contentType: "application/json; charset=utf-8",
        type: "POST",
        url: "/process_feedback_naive",
        data: JSON.stringify({'data' : data}),
        success: function(data, status){
            $('.images').empty();
            $(".images").html(data);
        },
    });     
    $('#count').html(iterationCount);
});
