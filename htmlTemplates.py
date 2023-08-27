css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 32px;
}
.chat-message .avatar img {
  max-width: 32px;
  max-height: 32px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  padding: 0 1.5rem;
  color: #fff;
}
.up_down {
    transform: rotate(180deg);
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" style="max-height: 32px; max-width: 32px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">
    {{MSG}}
    </div> 
    <a href="{{UPVOTE_URL}}"><img src="https://cdn-icons-png.flaticon.com/128/2107/2107671.png" style="width: 20px;height: 20px;" alt="Up Vote"></a>  
        <span style="width: 42px;"></span>
    <a href="{{DOWNVOTE_URL}}"><img src="https://cdn-icons-png.flaticon.com/128/2107/2107671.png" class="up_down" style="width: 20px;height: 20px;" alt="Down Vote"></a>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/rdZC7LZ/Photo-logo-1.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
