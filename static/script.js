let lightMode = true;
$("#light-dark-mode-switch").change(function () {
    $("body").toggleClass("dark-mode");
    $(".message-box").toggleClass("dark");
    $(".loading-dots").toggleClass("dark");
    $(".dot").toggleClass("dark-dot");
    lightMode = !lightMode;
  });


  async function showBotLoadingAnimation() {
    await sleep(200);
    $(".loading-animation")[1].style.display = "inline-block";
    document.getElementById('send-button').disabled = true;
  }
  function hideBotLoadingAnimation() {
    $(".loading-animation")[1].style.display = "none";
    if(!isFirstMessage){
        document.getElementById('send-button').disabled = false;
    }
  }

  async function showUserLoadingAnimation() {
    await sleep(100);
    $(".loading-animation")[0].style.display = "inline-block";

  }
function hideUserLoadingAnimation(){
    $(".loading-animation")[0].style.display = "none";
  
}
