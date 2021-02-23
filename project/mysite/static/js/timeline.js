$( document ).ready(function() {
    console.log( "ready!" );

    colorElements()
});

function colorElements(){
  var colors = ['#EC7063','#A569BD','#5DADE2','#45B39D','#58D68D','#F4D03F','#DC7633','#CC0099','#339900']
  var rows = $(".gantt-row-bars");
  var bars = $(rows).find('li')
  for (i = 0; i < bars.length; i++){
        $(bars[i]).css('background-color',colors[i%colors.length])
  }

}