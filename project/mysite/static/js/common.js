$( document ).ready(function() {
    console.log( "ready!" );

    window.onscroll = function() {
        progressBar();
        stickyNavbar();
    };

showSlide(0);
});
// Next/previous controls
function plusSlides(n) {
    var currentIndex = getCurrentIndex()
    var slides = $(".mySlides").length -1
    var nextIndex = currentIndex + n
    nextIndex = nextIndex > slides ? 0 : nextIndex < 0 ? slides : nextIndex;
    hideSlide(currentIndex)
    showSlide(nextIndex)
}

// Thumbnail image controls
function currentSlide(n) {
  var currentIndex = getCurrentIndex()
  hideSlide(currentIndex)
  showSlide(n)
}

function showSlide(n) {
  var slide = $(".mySlides")[n]
  var dot = $(".dot")[n]
  $(dot).addClass("active")
  $(slide).css("display","block")
}

function hideSlide(n) {
  var slide = $(".mySlides")[n]
  var dot = $(".dot")[n]
  $(dot).removeClass("active")
  $(slide).css("display","none")
}

function getCurrentIndex(){
  var visible = $(".mySlides:visible")
  var slides = $(".mySlides")
  var index = 0;
  for (i = 0; i < slides.length; i++){
        if($(slides[i]).is(visible))
            var index = i
  }
  return index
}

function stickyNavbar() {
    // Get the navbar
    var navbar = document.getElementById("navbar");
    // Get the offset position of the navbar
    var sticky = navbar.offsetTop;
    // Add the sticky class to the navbar when you reach its scroll position. Remove "sticky" when you leave the scroll position
  if (window.pageYOffset >= sticky) {
    navbar.classList.add("sticky")
  } else {
    navbar.classList.remove("sticky");
  }
}

function progressBar() {
  var winScroll = document.body.scrollTop || document.documentElement.scrollTop;
  var height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
  var scrolled = (winScroll / height) * 100;
  document.getElementById("progress-bar").style.width = scrolled + "%";
}

// Get cookie for CSRF token (from Django documentation)
function getCookie(name) {
  var cookieValue = null;
  if (document.cookie && document.cookie !== '') {
    var cookies = document.cookie.split(';');
    for (var i = 0; i < cookies.length; i++) {
      var cookie = jQuery.trim(cookies[i]);
      // Does this cookie string begin with the name we want?
      if (cookie.substring(0, name.length + 1) === (name + '=')) {
        cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
        break;
      }
    }
  }
  return cookieValue;
};

function loadAsync(url, id){
var csrftoken = getCookie('csrftoken');
$(id).hide()
$(id+'-loader').addClass('loader')
 $.ajax({
        type: 'POST',
        url: url,
        data: {
        'csrfmiddlewaretoken': csrftoken
        },
        success: function (data) {
        $(id).html(data);
        },
        error: function (response) {
        }
    }).done(function(){
        $(id+'-loader').removeClass('loader')
        $(id).show()
    })
}

function post_url(url){
var csrftoken = getCookie('csrftoken');
 $.ajax({
        type: 'POST',
        url: url,
        data: {
        'csrfmiddlewaretoken': csrftoken
        },
        success: function (data) {
            console.log(data)
        },
        error: function (response) {
            console.log("error")
        }
    })
}
